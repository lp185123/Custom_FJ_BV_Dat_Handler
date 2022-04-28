def missingXor(List1,partialList):
    xor_sum=0
    for num in List1:
        xor_sum^=num
    for num in partialList:
        xor_sum^=num
    return xor_sum


# define a list
my_list = [4, 7, 0, 3]

# get an iterator using iter()
my_iter = iter(my_list)

# iterate through it using next()

# Output: 4
print(next(my_iter))
print(next(my_iter))
print(next(my_iter))
print(zip(my_list,my_list))