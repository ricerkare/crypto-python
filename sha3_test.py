from sha3 import *
import random

capacities = [224, 256, 384, 512]
sha3_funcs = {
    224: sha3_224,
    256: sha3_256,
    384: sha3_384,
    512: sha3_512
}    

def do_tests(M):
    message = M.decode()
    if M == b"":
        message = "empty string"
    print("Message: " + message)
    for c in capacities:
        print(f"SHA3-{c}:", sha3_funcs[c](M))
    print("\n")

# Empty string.

M = ""
do_tests(M.encode())

# "abc".
M = "abc"
do_tests(M.encode())

# Random string of 100 characters.
M = ""
for i in range(100):
    c = int.to_bytes(random.randint(ord("a"), ord("z"))).decode()
    M += c
    
do_tests(M.encode())

# Random string of 1,000 lowercase characters.
M = ""
for i in range(1000):
    c = int.to_bytes(random.randint(ord("a"), ord("z"))).decode()
    M += c
    
do_tests(M.encode())

# Random string of 10,000 lowercase characters.
M = ""
for i in range(10000):
    c = int.to_bytes(random.randint(ord("a"), ord("z"))).decode()
    M += c
    
do_tests(M.encode())

# Edge cases - message length is 1 less than rate.

for c in capacities:
    M = "a"*(200 - 2*c//8 - 1)
    print("Message: " + M)
    print(f"SHA3-{c}:", sha3_funcs[c](M.encode()))
print("\n")
