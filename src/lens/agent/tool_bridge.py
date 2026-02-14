from __future__ import annotations

import json

from lens.agent.llm_client import ToolCall, ToolDefinition, ToolResult


def build_tool_definitions(adapter) -> list[ToolDefinition]:
    """Build agent tool definitions from a MemoryAdapter's capability manifest."""
    manifest = adapter.get_capabilities()

    # Build filter properties from manifest filter_fields
    filter_properties: dict = {}
    if hasattr(manifest, "filter_fields") and manifest.filter_fields:
        for ff in manifest.filter_fields:
            prop: dict = {"type": ff.field_type, "description": ff.description}
            if ff.enum_values:
                prop["enum"] = ff.enum_values
            filter_properties[ff.name] = prop

    # Describe available search modes and filter fields
    search_modes = ""
    if hasattr(manifest, "search_modes") and manifest.search_modes:
        search_modes = f" Available search modes: {', '.join(manifest.search_modes)}."

    filter_desc = ""
    if filter_properties:
        filter_desc = f" Available filter fields: {', '.join(filter_properties.keys())}."

    max_results = 10
    if hasattr(manifest, "max_results_per_search") and manifest.max_results_per_search:
        max_results = manifest.max_results_per_search

    # memory_search
    search_params: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query string.",
            },
            "limit": {
                "type": "integer",
                "description": f"Maximum number of results to return. Default: {max_results}.",
            },
        },
        "required": ["query"],
    }
    if filter_properties:
        search_params["properties"]["filters"] = {
            "type": "object",
            "description": "Optional filters to narrow search results.",
            "properties": filter_properties,
        }

    tools: list[ToolDefinition] = [
        ToolDefinition(
            name="memory_search",
            description=f"Search the memory system for relevant information.{search_modes}{filter_desc}",
            parameters=search_params,
        ),
        ToolDefinition(
            name="memory_retrieve",
            description="Retrieve a full document by its reference ID.",
            parameters={
                "type": "object",
                "properties": {
                    "ref_id": {
                        "type": "string",
                        "description": "The reference ID of the document to retrieve.",
                    },
                },
                "required": ["ref_id"],
            },
        ),
        ToolDefinition(
            name="memory_capabilities",
            description="Get the capability manifest describing what the memory system offers.",
            parameters={
                "type": "object",
                "properties": {},
            },
        ),
    ]

    # Add extra tools from manifest
    if hasattr(manifest, "extra_tools") and manifest.extra_tools:
        for et in manifest.extra_tools:
            tools.append(ToolDefinition(
                name=et.name,
                description=et.description,
                parameters=et.parameters,
            ))

    return tools


def dispatch_tool_call(
    adapter,
    tool_call: ToolCall,
    max_payload_bytes: int = 65536,
) -> ToolResult:
    """Route a tool call to the appropriate adapter method."""
    try:
        if tool_call.name == "memory_search":
            query = tool_call.arguments.get("query", "")
            filters = tool_call.arguments.get("filters")
            limit = tool_call.arguments.get("limit")
            results = adapter.search(query, filters, limit)
            payload = json.dumps(results, default=_serialize)
        elif tool_call.name == "memory_retrieve":
            ref_id = tool_call.arguments.get("ref_id", "")
            result = adapter.retrieve(ref_id)
            payload = json.dumps(result, default=_serialize)
        elif tool_call.name == "memory_capabilities":
            manifest = adapter.get_capabilities()
            payload = json.dumps(manifest, default=_serialize)
        else:
            result = adapter.call_extended_tool(tool_call.name, tool_call.arguments)
            payload = json.dumps(result, default=_serialize)

        # Truncate if needed
        payload_bytes = payload.encode("utf-8")
        if len(payload_bytes) > max_payload_bytes:
            truncated = payload_bytes[: max_payload_bytes - 20].decode("utf-8", errors="ignore")
            payload = truncated + "... [truncated]"

        return ToolResult(tool_call_id=tool_call.id, content=payload)

    except Exception as exc:
        return ToolResult(
            tool_call_id=tool_call.id,
            content=str(exc),
            is_error=True,
        )


def _serialize(obj: object) -> object:
    """Default serializer for json.dumps — handles objects with to_dict()."""
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        from dataclasses import asdict
        return asdict(obj)
    return str(obj)
