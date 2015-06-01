from pset2ex1 import Elems

print([e for e in Elems if e[0] == 's'])


def Elems_lessthan4():
    for e in Elems:
        if len(e) == 4:
            yield e
    raise StopIteration

print(list(Elems_lessthan4()))

# To be fair, a list comprehension is 'a generator expression in a list initialiser'
# but I still think these approaches demonstrate some (x)range.
