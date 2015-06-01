thermo = open('thermodynamics.txt', 'w+')

FirstLaw = """In all cases in which work is produced by the agency of heat,
 a quantity for heat is consumed which is proportional to the work done:
 and conversly, by the expenditure of an equal quantity of work and equal
 quantitiy for heat is produced"""

thermo.write(FirstLaw)

#This is caomment bro
thermo.seek(0)
ret = "The sixth word of this statment of the first law of thermodynamics is {}"
print(ret.format(thermo.read().split()[5]))
thermo.seek(0)
print(thermo.read())
