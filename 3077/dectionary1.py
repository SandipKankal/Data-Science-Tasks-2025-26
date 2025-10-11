'''Q.1) Write a Python program to create a lambda function that adds 15 to a given number passed in as an
argument.'''

'''b=lambda a: a+15
c=b(15)
print(c)'''

'''----------------------------------------------------------------------------------------------------------'''

'''Q.2) Write a Python program to sort a list of tuples using Lambda.
Original list of tuples:
[('English', 88), ('Science', 90), ('Maths', 97), ('Social sciences', 82)]

Sorting the List of Tuples:
[('Social sciences', 82), ('English', 88), ('Science', 90), ('Maths', 97)]

'''



'''----------------------------------------------------------------------------------------'''
'''Q.3) Write a Python program to filter a list of integers using Lambda.
Original list of integers:
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Even numbers from the said list:
[2, 4, 6, 8, 10]
Odd numbers from the said list:
[1, 3, 5, 7, 9]'''


'''def evenodd(a):
    if a%2==0:
       return True
    else:
       return False
a=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
filtered = list(filter(evenodd,a))
print(filtered)'''

'''def evenodd(a):
    if a%2==0:
       return False
    else:
       return True
a=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
filtered = list(filter(evenodd,a))
print(filtered)'''

'''----------------------------------------------------------------------------------------------------------'''

'''Q.4) Write a Python program to square and cube every number in a given list of integers using Lambda.
Original list of integers:
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Square every number of the said list:
[1, 4, 9, 16, 25, 36, 49, 64, 81, 100]
Cube every number of the said list:
[1, 8, 27, 64, 125, 216, 343, 512, 729, 1000]
'''
'''ls=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

a=lambda l : l*l
n=list(map(a,ls))
print(n)'''

'''ls=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

a=lambda l : l*l*l
n=list(map(a,ls))
print(n)'''
'''-------------------------------------------------------------------------------------------------------------'''

'''Q.5) Write a Python program to extract year, month, date from given date, using Lambda.
Sample Output:
2020-01-15

2020
1
15'''

'''------------------------------------------------------------------------------------------------------------'''

'''Q.6) Write a Python program to create Fibonacci series up to n using Lambda.
Fibonacci series upto 2:
[0, 1]
Fibonacci series upto 5:
[0, 1, 1, 2, 3]
Fibonacci series upto 6:
[0, 1, 1, 2, 3, 5]
Fibonacci series upto 9:
[0, 1, 1, 2, 3, 5, 8, 13, 21]
'''
'''fibonacci = lambda n: [] if n == 0 else [0] if n == 1 else [0, 1] if n == 2 else fibonacci(n - 1) + [fibonacci(n - 1)[-1] + fibonacci(n - 2)[-1]]

def print_fibonacci_series_up_to_n(n):
    series = fibonacci(n)
    print(f"Fibonacci series up to {n}: {series}")

# Test cases
print_fibonacci_series_up_to_n(2)
print_fibonacci_series_up_to_n(5)
print_fibonacci_series_up_to_n(6)
print_fibonacci_series_up_to_n(9)'''

'''-------------------------------------------------------------------------------------------------------------'''

'''Q.7) Write a Python program to find the intersection of two given arrays using Lambda.
Original arrays:
[1, 2, 3, 5, 7, 8, 9, 10]
[1, 2, 4, 8, 9]
Intersection of the said arrays: [1, 2, 8, 9]'''


'''a=[1, 2, 3, 5, 7, 8, 9, 10]
b=[1, 2, 4, 8, 9]
def intersect():
    
      c=[value for value in a if value in b] 
      print("intersection is ",c)

d=lambda :intersect()

d()'''

'''-----------------------------------------------------------------------------------------------------------'''

'''Q.8) Write a Python program to count the even and odd numbers in a given array of integers using Lambda.
Original arrays:
[1, 2, 3, 5, 7, 8, 9, 10]
Number of even numbers in the above array: 3
Number of odd numbers in the above array: 5'''

'''a=[1, 2, 3, 5, 7, 8, 9, 10]
even=0
odd=0
def count():
   global even,odd
   for i in range(7):
      if a[i]%2==0:
         even=even+1
      else:
         odd =odd +1
   print(even)
   print(odd)
    
d=lambda : count()
d()'''
'''-----------------------------------------------------------------------------------------------------------'''
'''Q.11) Write a Python program to add two given lists using map and lambda.
Original list:
[1, 2, 3]
[4, 5, 6]
Result: after adding two list
[5, 7, 9]'''

'''a=[1, 2, 3]
b=[4, 5, 6]

def add():
    for i in range(3):
        c=a[i]+b[i]
        print(c)
f=lambda:add()
f()'''

'''-------------------------------------------------------------------------------------------------------'''
'''.12) Write a Python program to find numbers divisible by nineteen or thirteen from a list of numbers using 
Lambda.
Orginal list:
[19, 65, 57, 39, 152, 639, 121, 44, 90, 190]
Numbers of the above list d
[19, 65, 57, 39, 152, 190]'''

a=[19, 65, 57, 39, 152, 639, 121, 44, 90, 190]
def div():
    global c
    for i in range(9):
      if  True:
         c=a[i]/13 or 19
   
print(c)
   
b= lambda:div()
b()
