P10 = ["hydrogen", "helium", "lithium", "beryllium", "boron", "carbon",
       "nitrogen", "oxygen", "fluorine", "neon"]
P20 = ["sodium", "magnesium", "aluminum", "silicon", "phosphorus", "sulfur",
       "chlorine", "argon", "potassium", "calcium"]
P30 = ["scandium", "titanium", "vanadium", "chromium", "manganese", "iron",
       "cobalt", "nickel", "copper", "zinc"]

Elems = P10 + P20
for e in P30:  # really
    Elems.append(e)

if __name__ == '__main__':
    print(len(Elems))
    print(Elems[22])
