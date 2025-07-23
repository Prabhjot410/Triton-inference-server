import redis

r = redis.Redis(
    host="redis-17045.c256.us-east-1-2.ec2.redns.redis-cloud.com",
    port=17045,
    decode_responses=True,
    username="default",
    password="rOidZZqUyDMlc8Q3cGMGvhmtWdyVfzjU"
)

# List all keys related to chats
keys = r.keys("chat:*")
print("Stored keys:")
for key in keys:
    print(key)

# Get and print values
for key in keys:
    print(f"\n== {key} ==")
    print(r.get(key))
