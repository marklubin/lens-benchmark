from kv_agent.lib.agent import Agent

# Initialize agent
agent = Agent()

# Test in history mode
agent.set_mode("history")
print("Testing in history mode:")
response = agent.respond("What is the capital of France?")
print(f"Response: {response}")

# Test KV store
print("\nKV Store contents:")
for item in agent.get_kv_store_contents():
    print(f"{item['key']}: {item['value']}")

# Test in kv_only mode
agent.set_mode("kv_only")
print("\nTesting in kv_only mode:")
response = agent.respond("What is the population of New York City?")
print(f"Response: {response}")

# Test KV store after second query
print("\nKV Store contents after second query:")
for item in agent.get_kv_store_contents():
    print(f"{item['key']}: {item['value']}")
