from collections import defaultdict

a1 = defaultdict(str)
a2 = defaultdict(str)
a3 = defaultdict(int)
a1["1"] = "abc"
# a1["2"] = "cde"
a2["1"] = "abc"
print(a1.__str__)
print(a2.__str__)
print(a1.get("3"))
print(a1.get("1"))
i = 0x000001F3003CB0B0
s = str(i)
a3[s] = i
print(a3) 

print(a1.__contains__("3"))
print(a1.__contains__("1"))
print(a1)
del a1["2"]
print(a1)