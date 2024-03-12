import math
import itertools
import collections
import random



"Ema's Supercomputer"

# # Input
# a = '''BBGGBGGBBGGGB
# BBGGBGGBBGGGB
# BBGGBGGBBGGGB
# BBGGBGGBBGGGB
# GGGGGGGGGGGGG
# BBGGBGGBBGGGB
# GGGGGGGGGGGGG
# BBGGBGGBBGGGB
# BBGGBGGBBGGGB
# GGGGGGGGGGGGG
# GGGGGGGGGGGGG
# GGGGGGGGGGGGG
# BBGGBGGBBGGGB
# GGGGGGGGGGGGG'''


# grid = [x for x in a.split('\n')]



# My solution

'1st Auxiliary Function'
# # a plus argument comes in the form "[Plus' area, (i, j)]"    being i,j the coordinates
# def plus_checker(plus):

#     if plus:

#         try:
#             area = plus[0]
#             up = plus[1][0]
#             down = plus[1][1]
#             left = plus[1][2]
#             right = plus[1][3]

#             group = ( grid[up[0]-1][up[1]], grid[down[0]+1][down[1]], grid[left[0]][left[1]-1], grid[right[0]][right[1]+1] )

#             within_limits = [up[0]>0, left[1]>0, down[0]<len(grid), right[1]<len(grid[0])]

#             if all(within_limits) and 'B' not in group:
#                 return [ area + 4,  ((up[0]-1, up[1]),  (down[0]+1, down[1]), (left[0], left[1]-1), (right[0], right[1]+1)) ] 


#         except IndexError:
#             return        



'2nd Auxiliary Function'
# # a plus argument comes in the form "[Plus' area, (i, j)]"    being i,j the coordinates
# def plus_overlap_check(pls1, pls2):

#     up1 = pls1[1][0]
#     down1 = pls1[1][1]
#     left1 = pls1[1][2]
#     right1 = pls1[1][3]

#     up2 = pls2[1][0]
#     down2 = pls2[1][1]
#     left2 = pls2[1][2]
#     right2 = pls2[1][3]

#     # This test will throw "False" when the pluses don't overlap
#     test_if_verts_meet = (up1[1] == up2[1]) and ( (up1[0] in range(up2[0], down2[0]+1)) or (down1[0] in range(up2[0], down2[0]+1)))
#     test_if_horizs_meet = (left1[0] == left2[0]) and ( (left1[1] in range(left2[1], right2[1]+1)) or (right1[1] in range(left2[1], right2[1]+1))) 
#     test_if_vrt1_meet_hrz2 = ( up1[1] in range(left2[1], down2[1]+1) ) and ( ( left1[0] in range(up2[0], down2[0]+1)) )

#     if any((test_if_verts_meet, test_if_horizs_meet, test_if_vrt1_meet_hrz2)):
#         return False
    
#     else:
#         return True



'Body of the main function'

# possible_p = list()

# for row in range(1, len(grid[1:-1])+1):
    
#     for j in range(1, len(grid[row][1:-1])+1):

#         if 'B' not in (grid[row-1][j], grid[row+1][j], grid[row][j-1], grid[row][j+1]):

#             # Stored as "[Plus' area, (the four ends of the plus)]"
#             possible_p.append([ 5, ( (row-1,j), (row+1,j), (row,j-1), (row,j+1) ) ])


# for i in range(len(possible_p)):

#     plus = possible_p[i]

#     while plus is not None:

#         plus = plus_checker(plus)

#         if plus is not None:

#             possible_p[i] = plus


# possible_p.sort(key= lambda x: x[0], reverse=True)


# for i in possible_p : print(i)
# print()


# if len(possible_p) == 0:
#     print( 0 )

# elif len(possible_p) == 1:
#     print( possible_p[0][0])

# else:
    
#     for i in range(len(possible_p)):
        
#         for j in range(len(possible_p[i+1:])):

#             if plus_overlap_check(possible_p[i], possible_p[j]) is True:

#                 print( possible_p[i][0] * possible_p[j][0])
            
#                 break
        
#         break






'plus_checker func tests cases'

    # test = [5, ((2,3), (4,3), (3,2), (3,4))]
    # grid = ['G'*(len(grid[0])+1)]*(len(grid)+1)

    # grid[1] = grid[1][:3] + 'B' + grid[1][3+1:]

    # for i in grid: print(i)

    # test = plus_checker(test)
    # print(test)

    # test = plus_checker(test)
    # print(test)

    # test = plus_checker(test)
    # print(test)

    # test = plus_checker(test)
    # print(test)

    # test = plus_checker(test)
    # print(test)




'''
My solution up to this point is 36% effective.

'''



# # Another guy's solution

# def twoPluses(grid):         
#     pluses = []  
#     for n in range(1, len(grid)-1):
#         for m in range(1, len(grid[0])-1):
#             if grid[n][m] == grid[n][m-1] == grid[n][m+1] == grid[n-1][m] == grid[n+1][m] == 'G':
#                 plus = [(n-1,m), (n,m-1), (n,m), (n,m+1), (n+1,m)]
#                 pluses.append(plus)
#                 i = 2
#                 while n+i < len(grid) and n-i >=0 and m+i < len(grid[0]) and m-i >= 0 and grid[n+i][m] == grid[n-i][m] == grid[n][m+i] == grid[n][m-i] == 'G':
#                     plus.extend([(n-i,m), (n,m-i), (n, m+i), (n+i,m)])
#                     pluses.append(sorted(plus))
#                     i += 1                
#     pluses.sort(key=lambda x: len(x), reverse=True)    
#     countG = 0
#     for row in grid:
#         countG += row.count('G') 
#     if len(pluses) == 0 and countG >= 2:
#         return 1
#     elif len(pluses) == 1:
#         if len(pluses[0]) == countG:
#             return 1
#         else:
#             return len(pluses[0])        
#     else:        
#         ans = len(pluses[0])        
#         for j,x in enumerate(pluses):
#             for y in pluses[j+1:]:
#                 for h,z in enumerate(x):
#                     if z in y:
#                         break
#                     elif h == len(x)-1 and len(x) * len(y) > ans:
#                         ans = len(x) * len(y)
#         return ans

"Solved with another guy's solution"






"Classes: Dealing with Complex Numbers"

# import math

# class Complex(object):

#     def __init__(self, real, imaginary):
#         self.real = real
#         self.imaginary = imaginary

#     def __add__(self, no):
#         return Complex(self.real+no.real, self.imaginary+no.imaginary)
        
#     def __sub__(self, no):
#         return Complex(self.real-no.real, self.imaginary-no.imaginary)
        
#     def __mul__(self, no):
        
#         new_real_part =  (self.real * no.real) - (self.imaginary * no.imaginary)
#         new_imaginary_part =  (self.real * no.imaginary) + (self.imaginary * no.real)
        
#         return Complex(new_real_part, new_imaginary_part)

#     def __truediv__(self, no):

#         new_real_part =  ( ((self.real * no.real) + (self.imaginary * no.imaginary)) / (pow(no.real, 2) + pow(no.imaginary, 2)) )
#         new_imaginary_part =  ( ((self.imaginary * no.real) - (self.real * no.imaginary)) / (pow(no.real, 2) + pow(no.imaginary, 2)) )
        
#         return Complex(new_real_part, new_imaginary_part)

#     def mod(self):
#         return Complex( math.sqrt( pow(self.real,2) + pow(self.imaginary,2) ) , 0 )

#     def __str__(self):

#         if self.imaginary == 0:
#             result = f"{self.real:.2f}+0.00i"

#         elif self.real == 0:

#             if self.imaginary >= 0:
#                 result = f"0.00+{self.imaginary:.2f}i"

#             else:
#                 result = f"0.00-{abs(self.imaginary):.2f}i"

#         elif self.imaginary > 0:       
#             result = f"{self.real:.2f}+{self.imaginary:.2f}i"

#         else:
#             result = f"{self.real:.2f}-{abs(self.imaginary):.2f}i"

#         return result


# A = complex(float(2),float(1))
# B = complex(float(5),float(6))

# C = Complex(float(2),float(1))
# D = Complex(float(5),float(6))

# print(D.mod())

# '-Classes: Dealing with Complex Numbers- HackerRank Problem Solving Challenge done'






"Re.start() & Re.end()"

# import re

# S = 'hbcasckjabchsdcsdcgvdsjhvcb'
# k = 'q'

# pattern = rf'(?=({k}))'

# matches = re.finditer(pattern, S)

# found_match = False

# for match in matches:

#     found_match = True
#     start_index = match.start(1)
#     end_index = match.end(1) - 1

#     print(f"({start_index}, {end_index})")

# if not found_match:    
#     print("(-1, -1)")



'''
    *Learned lesson:

        From this challenge I ran into something I did not know before:
        Iterators behave different than list in conditionals, because even
        when it has no matches it still exists, therefore it wouldn't be
        very useful to make a conditional like "if matches:" because it'd 
        never evaluate to false.

        The way around here is to use a flag to check whether there are matches in the iterator.
'''


# '-Re.start() & Re.end()- HackerRank Problem Solving Challenge done'






"Classes - Find the Torsional Angle"

# import math

# class Points(object):
    
#     def __init__(self, x, y, z):
        
#         self.x = x
#         self.y = y
#         self.z = z

#     def __sub__(self, no):
        
#         new_x = self.x - no.x
#         new_y = self.y - no.y
#         new_z = self.z - no.z
        
#         return Points(new_x, new_y, new_z)


#     def dot(self, no):
                       
#         return (self.x*no.x + self.y*no.y + self.z*no.z)
        
        
#     def cross(self, no):
        
#         new_x = (self.y * no.z) - (self.z * no.y)
#         new_y = (self.z * no.x) - (self.x * no.z)
#         new_z = (self.x * no.y) - (self.y * no.x)
        
#         return Points(new_x, new_y, new_z)
        
        
        
#     def absolute(self):
        
#         return pow((self.x ** 2 + self.y ** 2 + self.z ** 2), 0.5)



# if __name__ == '__main__':

#     points = list()
#     for i in range(4):
#         a = list(map(float, input().split()))
#         points.append(a)

#     a, b, c, d = Points(*points[0]), Points(*points[1]), Points(*points[2]), Points(*points[3])
#     x = (b - a).cross(c - b)
#     y = (c - b).cross(d - c)
#     angle = math.acos(x.dot(y) / (x.absolute() * y.absolute()))

#     print("%.2f" % math.degrees(angle))


# '-xxx- HackerRank Challenge done'




 