from sha3 import *
import random

def do_tests(M):
    message = M.decode()
    if M == b"":
        message = "empty string"
    print("Message: " + message)
    print("SHA3-224:", sha3_224(M))
    print("SHA3-256:", sha3_256(M))
    print("SHA3-384:", sha3_384(M))
    print("SHA3-512:", sha3_512(M))
    print("\n")

# EMPTY STRING

M = ""
do_tests(M.encode())

# "abc"

M = "abc"
do_tests(M.encode())

# Random string of 100 characters.
M = ""
for i in range(100):
    c = int.to_bytes(random.randint(ord("a"), ord("z"))).decode()
    M += c
    
do_tests(M.encode())

# Random string of 1,000 characters.
M = ""
for i in range(1000):
    c = int.to_bytes(random.randint(ord("a"), ord("z"))).decode()
    M += c
    
do_tests(M.encode())

# Random string of 100,000 characters.
M = ""
for i in range(100000):
    c = int.to_bytes(random.randint(ord("a"), ord("z"))).decode()
    M += c
    
do_tests(M.encode())
