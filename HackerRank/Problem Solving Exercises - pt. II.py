import math
import itertools
import collections
import random
import re


'Encryption'

# My Approach

# s = 'ifmanwasmeanttostayonthegroundgodwouldhavegivenusroots'
# # print(len(s))

# columns = math.ceil(math.sqrt(len(s)))
# # print(f'ceil: {columns}')

# rows = math.floor(math.sqrt(len(s)))
# # print(f'floor: {rows}')

# if rows * columns < len(s):
#     rows = columns

# result, rows_words = list(), list()

# for i in range(rows):
#     rows_words.append(s[i*columns:i*columns+columns])
    
# # print(rows_words)

# for i in range(columns):

#     temp = str()

#     for j in rows_words:

#         if i > len(j)-1:
#             continue
        
#         temp += j[i]
    
#     result.append(temp)

# # print(result)

# result_string = str()

# for i in result:    
#     result_string += i + ' '

# result_string.rstrip()

# print(result_string)


'''
This solution is the first draft that came to mind but I'm sure it could be a lot more efficient,
Below there is a solution that came from ChatGPT
'''

# def encryption(s):

#     s = s.replace(" ", "")  # Remove spaces
#     L = len(s)
#     R = int(math.floor(math.sqrt(L)))
#     C = int(math.ceil(math.sqrt(L)))
    
#     while R * C < L:
#         R += 1

#     grid = [['' for _ in range(C)] for _ in range(R)]

#     for i in range(L):
#         row = i // C
#         col = i % C
#         grid[row][col] = s[i]

#     encrypted_message = ""
#     for col in range(C):
#         for row in range(R):
#             if grid[row][col] != '':
#                 encrypted_message += grid[row][col]
#         encrypted_message += " "  # Add space between columns

#     return encrypted_message

# # Example usage:
# input_string = "ifmanwasmeanttostayonthegroundgodwouldhavegivenusroots"
# result = encryption(input_string)
# print(result)




'Bigger is Greater'


# w = 'dkhc'


# # Base solution
# comb = list(map(list, itertools.permutations(w)))
# comb = list(map(''.join,comb))

# comb.sort(reverse=True)
# ind = comb.index(w)
# result = comb[ind-1]

# print(comb)
# print(len(comb))
# print(result)


# # Solution sent to HackerRank
# def biggerIsGreater(w):
    
#     comb = list(map(list,itertools.permutations(w)))
#     comb = list(map(''.join,comb))
#     comb.sort(reverse=True)
#     ind = comb.index(w)
#     result = comb[ind-1]
    
#     if ind == 0 or result == w:
#         return 'no answer'
    
#     return result

''' 
This solution evidently is based on brute force,
3 out of the 5 cases failed due to time limit excedeed or runtime error,

    Pending: needs optimization.
'''


# ChatGPT's Solution

'''
According to ChatGPT here, the optimal solution which will be O(n), is to find the inmediate next permutation bigger
by finding the first pair (a,b) where a > b within the characters in the string, then find the rightmost character (c) which satisfies c > a, and then
swap those two and reverse that part of the string
'''

# def biggerIsGreater(w):
    
#     w = list(w)

#     i = len(w) - 2

#     while i >= 0 and w[i] >= w[i + 1]:
#         i -= 1

#     if i == -1:
#         return 'no answer'    


#     j = len(w) - 1

#     while w[j] <= w[i]:
#         j -= 1

#     w[i], w[j] = w[j], w[i]


#     w[i + 1:] = reversed(w[i + 1:])

#     return ''.join(w)

# print(biggerIsGreater(w))

'''
Done!
'''




'Modified Kaprekar Numbers'


# # My Solution
# p = 1
# q = 100

# rang = range(p,q+1)

# kapr_nums = list()

# for i in rang:

#     d = len(str(i))

#     pow2 = pow(i,2)
#     pow2 = str(pow2)
#     r = pow2[-d:]
#     l = pow2[:-d]

#     if l != '':
#         l,r = int(l),int(r) 
    
#     else:
#         l,r = 0, int(r)

#     if r + l == i:
#         kapr_nums.append(i)
        
# print(kapr_nums)


'''
My solution worked just fine, but I am curious if there was another way to do it without brut forcing it

    Note: After reviewing the discussion board and ChatGPTs opinion, all the solutions converge in the same solution.
'''




'Beutiful triplets'


# arr = [1, 2, 4, 5, 7, 8, 10]
# arr = [2,2,3,4,5]
# d = 3

# # First approach
# triplets = list(itertools.combinations(arr,3))

# b_triplets = list()

# for i in triplets:

#     if i[2] - i[1] == i[1] - i[0] == 3:
#         b_triplets.append(i)

# print(b_triplets)

'Brute forcing it wont do, Ill try somehting different'


# # Second approach
# b_triplets = list()

# for i in range(len(arr) - 2):

#     triplet = list()
#     triplet.append(arr[i])
    
#     for j in range(i+1, (len(arr)-2) + 1):

#         if arr[j] - arr[i] == d:
#             triplet.append(arr[j])

#             for k in range(j+1, len(arr)):

#                 if arr[k] - arr[j] == d:

#                     triplet.append(arr[k])
#                     b_triplets.append(triplet)
#                     triplet = triplet[:-1]
            
#             triplet = triplet[:-1]
  
# print(b_triplets)

'This second approach worked perfectly'




'Minimum Distances'


# # My Approach
# a = [3,2,1,2,3]
# a = [7, 1, 3, 4, 1, 7]

# distances = list()

# for ind, elem in enumerate(a):  

#     if a.count(elem) <= 1:
#         continue

#     for j in range(ind+1, len(a[ind+1:]) + 1):

#         if a[j] == elem:            
#             distances.append(abs(j-ind))

# distances.sort()

# print(distances)
# print(distances[0])

'''
When submitting my solution to HR, it worked 80% of the time
But after checking with ChtaGPT, there a couple of problems with my solution
to begin with it has a compl of O(n^2) and there is time when pairing are ignored
due to the 'range(ind+1, len(a[ind+1:]) + 1)' loop, 
and the final sorting adds another complexity of O(n log n), which, apparently is unnecessary

There's another approach based on dict that works more efficiently
'''

# a = [7, 1, 3, 4, 1, 7]


# def minimumDistance(a):

#     element_to_index = dict()
#     min_distance = float('inf')


#     for i, elem in enumerate(a):

#         if elem in element_to_index:

#             distance = i - element_to_index[elem]
#             min_distance = min(distance, min_distance)

#         element_to_index[elem] = i

#     return min_distance if min_distance != float('inf') else -1


# print(minimumDistance(a))


'This solution has a compl of O(n) and works 100% of times'




'Halloween Sale'

# inp = [int(x) for x in '100 19 1 180'.split()]

# p = inp[0]
# d = inp[1]
# m = inp[2]
# s = inp[3]

# count = int()

# if s >= p:

#     while s >= m and s >= p:

#         count += 1

#         if p > m:       
#             s -= p
#             p -= d

#         else:
#             s -= m

# else:
#     count = 0

'This solution worked just fine'




'The Time in Words'


# words = [
#     'one',
#     'two', 
#     'three', 
#     'four', 
#     'five', 
#     'six', 
#     'seven', 
#     'eight', 
#     'nine', 
#     'ten', 
#     'eleven', 
#     'twelve', 
#     'thirteen', 
#     'fourteen', 
#     'quarter', 
#     'sixteen', 
#     'seventeen', 
#     'eighteen', 
#     'nineteen', 
#     'twenty', 
#     'twenty one', 
#     'twenty two', 
#     'twenty three', 
#     'twenty four', 
#     'twenty five', 
#     'twenty six', 
#     'twenty seven', 
#     'twenty eight', 
#     'twenty nine', 
#     'half'
#     ]

# numbers = range(1,31)

# time_in_words = list(zip(numbers,words))

# dict_time = {k:v for k,v in time_in_words}


# h = 7
# m = 15

# if m == 0:
#     print(f"{dict_time[h]} o' clock")

# elif m == 1:
#     print(f"{dict_time[m]} minute past {dict_time[h]}")

# elif m == 15:
#     print(f"quarter past {dict_time[h]}")

# elif m == 30:
#     print(f"half past {dict_time[h]}")

# elif m > 30 and m != 45 and m != 59:
#     print(f"{dict_time[60-m]} minutes to {dict_time[h+1]}")

# elif m == 45:
#     print(f"quarter to {dict_time[h+1]}")

# elif m == 59:
#     print(f"one minute to {dict_time[h+1]}")

# else:
#     print(f"{dict_time[m]} minutes past {dict_time[h]}")


'Done!'




'Chocolate Feast'

# n = 15
# c = 3
# m = 2

# wrappers = int()
# chocolates = int()

# wrappers = chocolates = n // c

# while True:

#     chocolates += wrappers // m
#     wrappers = (wrappers // m) + (wrappers % m)

#     if wrappers < m:
#         break
    

# print(chocolates)

'Done!'




'Service Lane'

# width = [2, 3, 1, 2, 3, 2, 3, 3]
# cases = [[0, 3], [4, 6], [6, 7], [3, 5], [0, 7]]

# result = list()

# for case in cases:
#     result.append(min(width[case[0]:case[1]+1]))

# print(result)

'Done!'




"Lisa's Workbook"

# # Input
# n = 1
# k = 1
# arr = 100


# # My Approach
# page = 1
# workbook = list()
# special_problems = int()

# if type(arr) is int:
#     arr = [arr]

# for chapter in range(len(arr)):

#     problems_in_page = list()

#     for i in range(1, (math.ceil(arr[chapter]/k)*k)+1):
        
#         if i > arr[chapter]:
#             break

#         problems_in_page.append(i)

#         if len(problems_in_page) == k or i == arr[chapter]:         

#             workbook.append([page, problems_in_page.copy()])
#             page += 1            
#             problems_in_page.clear()
              

# for i in range(len(workbook)):

#     if workbook[i][0] in workbook[i][1]:

#         special_problems += 1
    

# print(workbook)
# print(special_problems)


'''
This solution worked, but seems little to forced. I am gonna check if there is something more sophisticated.

    There actually was but the logic was the same, there was a listcomp. to abreviate the separation of pages by chapter

    for problems_in_chapter in arr:
        chapter = [j for j in range(1, problems_in_chapter + 1)]
        chapter_splitted = [chapter[n : n + k] for n in range(0, len(chapter), k)]

but at the core, the solution was the same.

'''




"Flatland Space Stations"

# # Input
# n = 5
# c = [0, 4]


# # My approach
# cities = [city for city in range(n)]

# cities_with_nearest_st = list()
# max_distance = 0


# for city in cities:
    
#     dist_to_st = [abs(city - c[st]) for st in range(len(c))]

#     cities_with_nearest_st.append([city, min(dist_to_st)])



# for i in cities_with_nearest_st:

#     max_distance = max(max_distance, i[1])


# print(max_distance)


'My solution pass 19/20 cases and the one that didnt passed was due to time limit'


# # My second try
# cities = list(range(n))

# max_distance = 0

# for city in cities:
    
#     dist_to_st = [abs(city - c[st]) for st in range(len(c))]
#     max_distance = max(max_distance, min(dist_to_st))


# print(max_distance)

'After refactoring and cutting one loop, theres still not efficient enough'



# # ChatGPTs Solution

# def flatlandSpaceStations(n, c):

#     # Sort the list of space station locations
#     c.sort()
    
#     max_distance = 0
    
#     # Handle the first city to the first space station
#     max_distance = c[0]
    
#     # Handle the last city to the last space station
#     max_distance = max(max_distance, n - 1 - c[-1])
    
#     # Calculate the maximum distance between two adjacent space stations
#     for i in range(1, len(c)):
#         distance = (c[i] - c[i-1]) // 2
#         max_distance = max(max_distance, distance)
    
#     return max_distance

# print(flatlandSpaceStations(n, c))




"Fair Rations"

# B = [2, 3, 4, 5, 6]
# # B = [4,5,6,7]
# # B = [5,6,8,7]

# odds = [x for x in B if x%2 != 0]
# loaves = int()

# # print(odds)


# if len(odds)%2 != 0:
#     print('NO')
    

# else:
#     for i in range(len(B)):

#         if B[i]%2 == 0:
#             continue

#         else:            
#             B[i] += 1 
#             B[i+1] += 1
#             loaves += 2


# print(loaves)

'This one went smoothly, Done!'




"Cavity Map"        

# grid = ['1112', '1912', '1892', '1234']

# # 1 1 1 2
# # 1 9 1 2
# # 1 8 9 2
# # 1 2 3 4

# grid = [[int(x) for x in elem] for elem in grid]
# cavities = list()

# for row in range(1, len(grid)-1):

#     for col in range(1, len(grid[row])-1):

#         cell = grid[row][col]
#         group = (grid[row][col-1], grid[row][col+1], grid[row-1][col], grid[row+1][col])
        
#         if 'X' not in group and cell > max(group):
#             grid[row][col] = 'X'

#         else:
#             continue

# grid = [''.join(str(x) for x in row) for row in grid]

# for i in grid:
#     print(i)

'This one also went smoothly, Done!'




"Manasa and Stones"

# n = 3
# a = 1
# b = 2

# possibilities = list(itertools.combinations_with_replacement((a,b),r=n-1))

# print(possibilities)
# print()

# possibilities = sorted(map(sum, possibilities))

# result = []

# [result.append(x) for x in possibilities if x not in result]

# print(result)
# print()

'Done!'




"The Grid Search"

# # My approach:
# def index_caller(li, val):

#     occur = list()

#     try:
#         while True:
#             index = li.index(val)
#             occur.append(index)
#             li = li[index+1:]
    
#     except ValueError:
#         pass

#     return occur


# G = ['7283455864', '6731158619', '8988242643', '3830589324', '2229505813', '5633845374', '6473530293', '7053106601', '0834282956', '4607924137']
# P = ['9505', '3845', '3530']


# # 7 2 8 3 4 5 5 8 6 4
# # 6 7 3 1 1 5 8 6 1 9
# # 8 9 8 8 2 4 2 6 4 3
# # 3 8 3 0 5 8 9 3 2 4
# # 2 2 2 9 5 0 5 8 1 3
# # 5 6 3 3 8 4 5 3 7 4
# # 6 4 7 3 5 3 0 2 9 3
# # 7 0 5 3 1 0 6 6 0 1
# # 0 8 3 4 2 8 2 9 5 6
# # 4 6 0 7 9 2 4 1 3 7
# # G = [ [int(x) for x in elem] for elem in G ]

# # 9 5 0 5
# # 3 8 4 5
# # 3 5 3 0
# # P = [ [int(x) for x in elem] for elem in P ]

# res = 'NO'

# for i in range(len(G)-len(P) + 1):    
            
#     if P[0][0] in G[i][: len(G[0])-len(P[0]) + 1 ]:

#         occur = [i for i,v in enumerate(G[i][: len(G[0])-len(P[0]) + 1 ]) if v == P[0][0]]
#         # occur = index_caller(G[i][: len(G[0])-len(P[0]) + 1 ], P[0][0] )

#         for j in occur:

#             sample = [G[i+x][j:j+len(P[0])] for x in range(len(P))]

#             if sample == P:
#                 print('Found It!')
#                 break


'''
    While I thought my solution without the auxiliary function 'index_caller' may be a good idea
    turns out, my first solution with the indexes called by a listcomp was better, actually worked 94% of times
    and the other 6% was due to time exceeding the limits, so to go on, optimizations must be made:

        - I would start by checking if there's anything better than the two comprehensions at the begining.


        I got a Hint: what if there is not need to turn the elements into integer but to compare them as strings directly?
'''

'Welp, apparently that was it, it was just a matter of not turning them into strings'




"Happy Ladybugs"

# b = 'G'

# # Counter({'_': 3, 'Q': 3, 'D': 2, 'F': 2})
# dic = collections.Counter(b)

# def already_happy(string):

#     test = list()
#     if len(string) > 1:
#         for i, v in enumerate(string):

#             if v != '_':
#                 if i == 0:
#                     test.append(True) if string[i+1] == v else test.append(False)

#                 elif i == len(string)-1:
#                     test.append(True) if string[i-1] == v else test.append(False)

#                 else:
#                     test.append(True) if string[i+1] == v or string[i-1] == v else test.append(False)
            
#         return all(test)


# if already_happy(b):
#     print('YES')

# else:

#     lady_bugs_with_more_than_one = [x for x in dic if dic[x] != 1 and x != '_']
#     dict_wo_undr = [x for x in dic if x != '_'] 
    
#     if len(lady_bugs_with_more_than_one) == len(dict_wo_undr) and dic['_'] > 0:
#         print('YES')
    
#     else:
#         print('NO')




# # Case Generator to observe
# #   Case 1: RYR_YYXYRX
# a = ['X','R','Y', '_']
# li = ''.join([a[random.randint(0,3)] for _ in range(8)])

# print(li)

'''
Observations

    No singletons: characters different from '_'   /:   if count(char) that is non '_' is == 1, NO
    No already arrange without any spaces (no '_')   /:   if no '_' and any char is not at side of the same letter, NO

    if all '_', YES

'''

'It worked! Done!'




"Strange Counter"

# t = 22

# time = 1
# value = 3

# last_value = int()
# last_time = int()


# '3'
# while time < t:
#     last_value = value
#     value *= 2
#     last_time = time
#     time += value - last_value


# if time == t:
#     print(value)

# else:
#     result = last_value - (t-last_time)
#     print(result)


'Done!'




"Absolute Permutation"

# n = 10
# k = 0

# def absolutePermutation(n, k):
    
#     if k == 0:
#         return [x for x in range(1, n+1)]

#     elif n % (2*k) != 0:
#         return -1
    
#     else: 
#         abs_permutation = list()
        
#         for i in range(0, int(n/(2*k))):
            
#             for j in range(1, (2*k)+1):

#                 if j <= k:
#                     abs_permutation.append(((i*(k*2))+j)+k)
#                 else:
#                     abs_permutation.append(((i*(k*2))+j)-k)
        
#         return abs_permutation


# print(absolutePermutation(n,k))




"The Bomberman Game"



# Inputs

n = 5

a = '''.......
...O.O.
....O..
..O....
OO...OO
OO.O...'''

grid = [x for x in a.split('\n')]






# My Approach

def bombs_effect_map(grid):
    
    bombs_effect = list()

    for i in range(len(grid)):

        for j in range(len(grid[i])):

            if grid[i][j] == 'O':
                
                #Top three cases: up-left corner, up mid, up-right corner
                if i == 0 and j == 0:
                    bombs_effect.extend([(i,j), (i+1,j), (i,j+1)])
                
                elif i == 0 and j < len(grid[0])-1:
                    bombs_effect.extend([(i,j), (i+1,j), (i,j+1), (i,j-1)])
                
                elif i == 0 and j == len(grid[0])-1:
                    bombs_effect.extend([(i,j), (i+1,j), (i,j-1)])


                #Middle three cases: mid-left corner, mid-mid, mid-right
                elif i < len(grid)-1 and j == 0:
                    bombs_effect.extend([(i,j), (i+1,j), (i-1,j), (i,j+1)])
                
                elif i < len(grid)-1 and j < len(grid[0])-1:
                    bombs_effect.extend([(i,j), (i-1,j), (i+1,j), (i,j+1), (i,j-1)])

                elif i < len(grid)-1 and j == len(grid[0])-1:
                    bombs_effect.extend([(i,j), (i-1,j), (i+1,j), (i,j-1)])



                #Bottom three cases: down-left corner, down-mid, down-right
                elif i == len(grid)-1 and j == 0:
                    bombs_effect.extend([(i,j), (i-1,j), (i,j+1)])
                
                elif i == len(grid)-1 and j < len(grid[0])-1:
                    bombs_effect.extend([(i,j), (i-1,j), (i,j+1), (i,j-1)])
                
                elif i == len(grid)-1 and j == len(grid[0])-1:
                    bombs_effect.extend([(i,j), (i-1,j), (i,j-1)])
                

    grid = ['.' * len(grid[0])] * len(grid)

    for i in range(len(grid)):

        for j in range(len(grid[i])):

            if (i,j) not in bombs_effect:
                grid[i] = grid[i][:j]+'O'+grid[i][j+1:]

    return grid       


def bomberMan(n, grid):

    if n == 0:
        return grid
    
    
    elif n % 3 == 1:

        if n % 2 == 0:
            return ['O'*len(grid[0])]*len(grid)
        
        else:
            if n % 12 == 1:
                return bombs_effect_map(bombs_effect_map(grid))
            
            elif n % 12 == 7:
                return bombs_effect_map(grid)
    

    elif n % 3 == 0:
         
        if n % 2 == 0:
            return ['O'*len(grid[0])]*len(grid)
        
        else:
            if n % 12 == 3:
                return bombs_effect_map(grid)
            
            elif n % 12 == 9:
                return bombs_effect_map(bombs_effect_map(grid))
            

    elif n % 3 == 2:
         
        if n % 2 == 0:
            return ['O'*len(grid[0])]*len(grid)
        
        else:
            if n % 12 == 5:
                return bombs_effect_map(bombs_effect_map(grid))
            
            elif n % 12 == 11:
                return bombs_effect_map(grid)




for i in bomberMan(n, grid):
    print(i)


# # # Reasoning to identify the inner conditionals' condition
# # # mod 1
# # a = [1,7,13,19,25,31,37,43]  # Result mod12: [1, 7, 1, 7, 1, 7, 1, 7]
# # 1: 2xBomberman / 7: BomberMan
# # #mod 0
# # b = [3,9,15,21,27,33,39,45,51]  # Result mod12: [3, 9, 3, 9, 3, 9, 3, 9, 3]
# # 3: Bomberman / 9: 2xBomberMan
# # #mod 2
# # c = [5,11,17,23,29,35,41,47,53]  # Result mod12: [5, 11, 5, 11, 5, 11, 5, 11, 5] 
# # 5: 2xBomberman / 11: BomberMan

# # print(list(map(lambda x: x%12, c)))







'''
At my best, my solution works 46% of the time and I for real ran out of energy at this point, fuck it!
Tomorrow I am going to study another solution for this problem.

    PS: I also think the way the problem was redacted does not give much and lacks clarity to solve it properly.

    20 Oct - Update: I made a misobservation leading me to give the incorrect output, after correcting the timing the efficacy of my solution
    grew 10%, now being 56% of the time giving the right output in the right time. Now the solution does not run fast enough and that's the rest 
    of the 44% of cases.

'''


print()

# # Another Approach
# def bombs_list(grid):
#     bombs = []
#     for i,x in enumerate(grid):        
#         bombs.extend([(i, m.start(0)) for m in re.finditer(chr(79), x)])
#     return bombs

# def bomberMan2(n, grid):
#     # Write your code here
#     if n == 1:
#         return grid
        
#     full_row = chr(79) * len(grid[0])
#     full_grid = [full_row] * len(grid)
        
#     if n % 2 == 0:        
#         return full_grid
             
#     bombs = bombs_list(grid)
#     for _ in range(2):        
#         gb = full_grid.copy()        
#         for j in bombs:
#             if j[0]-1 >= 0:                
#                 gb[j[0]-1] = gb[j[0]-1][:j[1]] + '.' + gb[j[0]-1][j[1]+1:]
#             if j[0]+1 < len(grid):                
#                 gb[j[0]+1] = gb[j[0]+1][:j[1]] + '.' + gb[j[0]+1][j[1]+1:]
#             if j[1]+1 < len(grid[0]):                
#                 gb[j[0]] = gb[j[0]][:j[1]+1] + '.' + gb[j[0]][j[1]+2:]
#             if j[1]-1 >= 0:                
#                 gb[j[0]] = gb[j[0]][:j[1]-1] + '.' + gb[j[0]][j[1]:]
#             gb[j[0]] = gb[j[0]][:j[1]] + '.' + gb[j[0]][j[1]+1:]
#         if n == 3 or (n-3)%4 == 0:
#             return gb
#         bombs = bombs_list(gb)          
#     return gb


# for i in bomberMan2(n, grid):
#     print(i)

