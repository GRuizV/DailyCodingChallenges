from timeit import default_timer as timer

'''
Subarray Division

Two children, Lily and Ron, want to share a chocolate bar. Each of the squares has an integer on it.

Lily decides to share a contiguous segment of the bar selected such that:

The length of the segment matches Ron's birth month, and,
The sum of the integers on the squares is equal to his birth day.
Determine how many ways she can divide the chocolate.

'''


# def birthday(s, d, m):

#     res = list()

#     for i in range((len(s) + 1 - m)):
        
#         temp = list()

#         for j in range(m):

#             temp.append((s[i+j]))
        
#         res.append(temp)

#     res = [x for x in res if sum(x) == d]

#     return len(res)



'Test of the func'

# s = [1, 2, 1, 3, 2]

# m = 2

# d = 3

# res = list()

# for i in range((len(s) + 1 - m)):

#     temp = list()

#     for j in range(m):

#         temp.append(s[i+j])

#     res.append(temp)

# res = [x for x in res if sum(x) == d]

# print(res)




'''
Divisible Sum Pairs

Given an array of integers and a positive integer k, 
determine the number of (i,j) pairs where i<j and ar[i] + ar[j] is divisible by k.

constrains

i < j

(i+j) % k == 0


Example

ar = [1,2,2,4,5,6]

k = 5

three pais meet the criteria: [1,4], [2,3] and [4,6]

'''

# ar = [1,2,3,4,5,6]

# k = 5


# pairs = list()

# for i in range(len(ar)):

#     for j in range(len(ar)):

#         if i != j and i < j and (ar[i] + ar[j]) % k == 0: 

#             pairs.append((ar[i], ar[j]))
    

# print(pairs)




'''
Day of the Programmer

Complete the dayOfProgrammer function in the editor below. 
It should return a string representing the date of the 256th day of the year given.


In the Julian calendar:
- leap years are divisible by 4;

if 1918 -> Feb 14th

in the Gregorian calendar, leap years are either of the following:
- Divisible by 400.
- Divisible by 4 and not divisible by 100.


# non LpY 2023-09-13
# LpY 2000-09-12
# if 1918 -> 2000-08-31


'''


# year = 2400

# pr_day = 256


# res = int()

# if year <= 1917:

#     if year % 4 == 0:

#         print(f'12.09.{year}')

#     else:

#         print(f'13.09.{year}')

# elif year == 1918:

#         print(f'31.08.{year}')

# else:
     
#     if year % 4 == 0 and year % 100 != 0 or year % 400 == 0:

#         print(f'12.09.{year}')

#     else:

#         print(f'13.09.{year}')
         

 
'''
INPUT FOR HR

def dayOfProgrammer(year):
    
    if year <= 1917:

        if year % 4 == 0:

            return f'12.09.{year}'

        else:

            return f'13.09.{year}'

    elif year == 1918:

        return f'26.09.{year}'

    else:
     
        if year % 4 == 0 and year % 100 != 0 or year % 400 == 0:

            return f'12.09.{year}'

        else:

            return f'13.09.{year}' 
 
'''






'''
Drawing Book

'''

# from math import floor, ceil

# n = 6

# p = 3



# if p <= n/2: 

#     print(f'{ceil((p-1)/2)}')

# else:

#     if n%2 == 0:

#         print(f'{ceil((n-p)/2)}')
    
#     else:
        
#         print(f'{floor((n-p)/2)}')






'''
Counting Valleys
'''

# path = 'DDUUDDUDUUUD'

# level, valleys, prev_lvl = int(), int(), int()

# for n, e in enumerate(path):

#     if e == 'U':

#         level += 1

#     elif e == 'D':

#         level -= 1

#     if prev_lvl < 0 and level == 0:

#         valleys += 1

#     prev_lvl = level

      
# print(valleys)






'''
Electronic Shop
'''


# b = 10

# k = [3, 1]

# d = [5, 2, 8]

# res = sorted([x+y for x in k for y in d if x+y <= b], reverse= True)


# print(res)






'''
Forming a Magic Square
'''

# def square_check(m):

#     magic_squares = [

#         [6, 1, 8, 7, 5, 3, 2, 9, 4], #0
#         [8, 1, 6, 3, 5, 7, 4, 9, 2], #1
#         [6, 7, 2, 1, 5, 9, 8, 3, 4], #2
#         [8, 3, 4, 1, 5, 9, 6, 7, 2], #3
#         [2, 7, 6, 9, 5, 1, 4, 3, 8], #4
#         [4, 3, 8, 9, 5, 1, 2, 7, 6], #5
#         [2, 9, 4, 7, 5, 3, 6, 1, 8], #6
#         [4, 9, 2, 3, 5, 7, 8, 1, 6], #7

#     ]

#     costs = list()

#     for n,i in enumerate(magic_squares):

#         cost = sum( [ abs(x[0] - x[1] ) for x in zip(i,m)  ]  )

#         costs.append((n, cost))

#     print(sorted(costs, key= lambda x: x[1], reverse= False))


# #m = [[5, 3, 4], [1, 5, 8], [6, 4, 2]]

# #m = [[4, 9, 2], [3, 5, 7], [8, 1, 5]]

# m = [[4, 8, 2], [4, 5, 7], [6, 1, 6]]

# m = [x for y in m for x in y]

# square_check(m)






'''
Picking Numbers
'''


# l = [1, 2, 2, 3, 1, 2, 3, 4, 4, 3, 2, 5, 4, 6]

# l = [1, 2, 2, 3, 1, 2]




'This effort was a lot and works most of the times but at the end failed with large groups'

# subsets = list()


# for i in l:
    
#     temp = tuple(x for x in l if abs(x-i) <= 1)

#     if temp in subsets:
#         continue

#     else:
#         subsets.append(temp)


# print(subsets)

# print()


# remove_subs = list()

# for i in subsets:

#     temp = list()

#     for j in i:

#         temp.append( list( map(lambda x: abs(j-x), i) ) )
    

#     # print(temp)


#     for elem in temp:

#         if all(diff <= 1 for diff in elem) == False:

#             if i in remove_subs:
#                 continue

#             else: 

#                 remove_subs.append(i)

       
# print(remove_subs)


# subsets = sorted([x for x in subsets if x not in remove_subs], key=len, reverse=True)

# print()

# print(subsets)

# print( len(subsets[0]) )



' This solution came from Youtube '


# from collections import Counter


# count = Counter(l)

# print(count)

# maxnum = 0

# for i in range(1, 100):

#     maxnum = max(maxnum, count[i] + count[i+1]) #This Idea of saving the max between two by filtering with the built-in function is brilliant

# print(maxnum)






'''
Climbing the Leaderboard

'''

# ranked = [100,100,50,40,40,20,10]

# player = [5,25,50,120]


# # ranked, player = ranked.split(), player.split()

# ranked = sorted(set(ranked))

# print(ranked)

# # ranked = [10, 20, 40, 50, 100] / len = 5
# n = len(ranked)

# i = 0

# player_ranks = list()


# for score in player:

#     #This loop makes sure to go thru the ranked scores only once, which time saves.
#     while i < n and ranked[i] <= score:

#         i += 1

#     player_ranks.append(n - i + 1)
    


# print(player_ranks)






'''
Designer PDF Viewer

'''

# import string

# letters = 'abcdefghijklmnopqrstuvwxyz'

# h = [1, 3, 1, 3, 1, 4, 1, 3, 2, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

# heights = {k:v for k,v in zip(letters,h)}

# word = 'abc'

# width = len(word)

# max_height = max([heights[i] for i in word])

# print(width*max_height)




'''
Utopian Tree

'''

# n = 7

# heights = [1]


# for i in range(1,n+1):

#     if i%2 != 0:

#         heights.append(heights[i-1]*2)

#     else:

#         heights.append(heights[i-1]+1)


# # #listcomp

# # l = [ heights.append(heights[i-1]*2) if i%2 != 0 else heights.append(heights[i-1]+1) for i in range(1,n+1) ]

# print(heights)






'''
Sequence Equation
'''

# l = '4 3 5 1 2'

# l = list(map(int,l.split()))

# res = list()

# for i in range(1, len(l)+1):

#     x = l.index(i)+1

#     px = l.index(x)+1

#     res.append(px)


# # [4, 5, 2, 1, 3]
# print( res)






'''
Jumping on the Clouds: Revisited
'''

# Cases

# '0 0 1 0 0 1 1 0' # n=8 k=2
# '1 1 1 0 1 1 0 0 0 0' # n=10 k=3
# '1 1 0 1 0 1 0 1 0 1 0 1 1 0 1 1 1 1 1' # n=19 k=19



'--------------------------'


'This way is the chambonazo one'

# s = timer()
# thunders, jumps, = int(), int()

# c = list( map( int, '1 1 1 0 1 1 0 0 0 0'.split()))

# n = len(c)
# k = 3
# ini_c = c[0]
# c[0] = 'x'
# jumps = list()

# 'here goes a condition for the special case k = len(c) and return either 99 or 97 depending wether the c[0] is 1 or 0?'
# i = k

# while c[i] != 'x': 

#     jumps.append(c[i])
    
#     i = (i+k) % n


# if ini_c == 0:

#     e = 100 - (len(jumps)+1) - (len([x for x in jumps if x == 1])*2)
#     print(e)

# else:

#     e = 100 - (len(jumps)+1) - ( len([x for x in jumps if x == 1]) + 1 )*2
#     print(e)

# e = timer()

# print(e-s)

# print()


'--------------'


'I know I can do this better'



# s = timer()

# c = list( map( int, '1 1 1 0 1 1 0 0 0 0'.split()))

# k = 3

# def jumping_clouds(c,k):

#     n = len(c)

#     jumps, thunders = int(), int()

#     if k == n:
        
#         if c[0] == 0:
#             return 99
        
#         else: 
#             return 97
        
#     elif n % k == 0:

#         jumps = int(n/k)

#         thunders = len([ c[x] for x in range(0, n, k) if c[x] == 1])        
    
#     else:

#         jumps = n

#         thunders = len([x for x in c if x == 1])

#     return 100 - jumps - thunders*2

# res = jumping_clouds(c, k)

# print(res)


# e = timer()

# print(e-s)


'Done, and this approach is more efficient, actually twice more efficient'






'''
Empty String
'''

# 1. determine until which point the two strings are similar
# 2. determine the variation by characters
# 3. if there are more characters than K, return 'No' else 'Yes'


# s = 'y'

# t = 'yu'

# k = 2



# res, ind = str(), str()

# iter_len = int()


# if len(s) > len(t): 
#     iter_len = len(t)
# else:
#     iter_len = len(s)


# for i in range(iter_len):

#     if s[i] == t[i]:
#         res += s[i]
#         ind = i

#     else:
#         break


# chars_to_change = s[ind+1:] + t[ind+1:]

# # print(res)

# if len(chars_to_change) > k or len(s) < len(t):
#     print('No')

# else:
#     print('Yes')






'''
Cut the sticks
'''
 
# arr = [5, 4, 4, 2, 2, 8]

# # Expected result: [6, 4, 2, 1]

# res = list()

# while len(arr) != 0:

#     res.append(len(arr))

#     cut = min(arr)
    
#     arr = [x for x in [x-cut for x in arr] if x > 0]

# print(res)






'''
Non-Divisible Subset
'''

# import itertools

# l = [int(x) for x in '1 7 2 4'.split()] 

# k = 3

# # [4, 3, 2, 1]
# sizes = list(range(len(l), 0, -1))

# result = 0


# for i in sizes:
    
#     subset = list(itertools.combinations(sorted(l), i))
#     # print(subset)
     
#     for j in subset:

#         permutations = list(itertools.combinations(j, 2))

#         test = [True if sum(x)%k == 0 else False for x in permutations] 

#         if any(test):
#             continue

#         else:
#             result = i
#             break
    
#     if result != 0:
#         break


# print(result)





_ = 'Equalizing the Array'


# arr = [10, 27, 9, 10, 100, 38, 30, 32, 45, 29, 27, 29, 32, 38, 32, 38, 14, 38, 29, 30, 63, 29, 63, 91, 54, 10, 63]

# d = sorted({k:arr.count(k) for k in arr}.items(), key = lambda x: x[1], reverse = True)

# max_count_item = d[0][0]

# print(len(arr)-arr.count(max_count_item))





_ = "Queen's Attack"


# # n = 5

# # k = 3


# # # (4,3)
# # queen_pos = (4,3)



# # queen_attacks = list()


# # #First: the Cross section

# # # Horizontal
# # [queen_attacks.append((queen_pos[0], x)) for x in range(1,n+1) if (queen_pos[0], x) != queen_pos ]

# # # Vertical
# # [queen_attacks.append((x, queen_pos[1])) for x in range(1,n+1) if (x, queen_pos[1]) != queen_pos ]



# # #Second: left to right diagonal
# # # up-part
# # [queen_attacks.append((queen_pos[0]+x, queen_pos[1]+x)) for x in range(n-queen_pos[0]+1) if (queen_pos[0]+x, queen_pos[1]+x) != queen_pos and queen_pos[0]+x <= n and queen_pos[1]+x <= n ]

# # # down-part
# # [queen_attacks.append((queen_pos[0]-x, queen_pos[1]-x)) for x in range(1,queen_pos[0]) if (queen_pos[0]-x, queen_pos[1]-x) != queen_pos and queen_pos[0]-x >= 1 and queen_pos[1]-x >= 1 ]



# # #Third: right to left diagonal
# # # up-part
# # [queen_attacks.append((queen_pos[0]+x, queen_pos[1]-x)) for x in range(n-queen_pos[0]+1) if (queen_pos[0]+x, queen_pos[1]+x) != queen_pos and queen_pos[0]+x <= n and queen_pos[1]-x >= 1 ]

# # # down-part
# # [queen_attacks.append((queen_pos[0]-x, queen_pos[1]+x)) for x in range(1,queen_pos[0]) if (queen_pos[0]-x, queen_pos[1]-x) != queen_pos and queen_pos[0]-x >= 1 and queen_pos[1]+x <= n ]



# # # [(5,5), (4,2), (2,3)]
# # obstacles = [(5,5), (4,2), (2,3)]
# # obstacles = list(set(obstacles))




# # obstacle_bloq = list()


# # for i in obstacles:

# #     if i in queen_attacks:

# #         # Horizontal bloqueages
# #         if i[0] == queen_pos[0]: 
            
# #             if i[1] > queen_pos[1]:                
# #                 [ obstacle_bloq.append( (i[0],x)) for x in range(i[1], n+1) ]
            
# #             else:
# #                 [ obstacle_bloq.append((i[0],x)) for x in range(i[1], 0, -1) ]

# #         # Vertical bloqueages
# #         elif i[1] == queen_pos[1]: 
            
# #             if i[0] > queen_pos[0]:
# #                 [ obstacle_bloq.append( (x, i[1])) for x in range(i[0], n+1) ]
            
# #             else:
# #                 [ obstacle_bloq.append( (x, i[1])) for x in range(i[0], 0, -1) ]


# #         # Diagonal bloqueages
# #         else: 
            
# #             # from left to right up
# #             if i[0] > queen_pos[0] and i[1] > queen_pos[1]:
# #                 [ obstacle_bloq.append( (i[0]+x, i[1]+x)) for x in range(n-i[0]+1) if i[0]+x <= n and i[1]+x <= n ]

# #             # from left to right down
# #             if i[0] > queen_pos[0] and i[1] < queen_pos[1]:
# #                 [ obstacle_bloq.append( (i[0]+x, i[1]-x)) for x in range(n-i[0]+1) if i[0]+x <= n and i[1]-x > 0 ]
            
# #             # from right to left up
# #             if i[0] < queen_pos[0] and i[1] > queen_pos[1]:
# #                 [ obstacle_bloq.append( (i[0]-x, i[1]+x)) for x in range(n-i[1]+1) if i[0]-x > 0 and i[1]+x <= n ]

# #             # from right to left down
# #             if i[0] < queen_pos[0] and i[1] < queen_pos[1]:
# #                 [ obstacle_bloq.append( (i[0]-x, i[1]-x)) for x in range(n-i[1]+1) if i[0]-x > 0 and i[1]-x > 0 ]



# # print(obstacle_bloq)

# # result = set(queen_attacks).difference(set(obstacle_bloq))

# # print(queen_attacks)

# # print(len(result))  




# _ = 'The code works well but not scalable, needs adjustments on efficiency / I Will try a different Approach'
# 'Another approach in HackerRank is the following'



# n = 5

# k = 3

# queen_pos = (4,3)

# obstacles = [(5,5), (4,2), (2,3)]



# ul_max = min(queen_pos[1]-1, n-queen_pos[0])
# ur_max = min(n-queen_pos[1], n-queen_pos[0])
# dl_max = min(queen_pos[1]-1, queen_pos[0]-1)
# dr_max = min(n-queen_pos[1], queen_pos[0]-1)
# l_max = queen_pos[1]-1
# r_max = n-queen_pos[1]
# u_max = n-queen_pos[0]
# d_max = queen_pos[0]-1


# for obs in obstacles:

#     [obs_r, obs_c] = obs

#     if obs_r + obs_c == queen_pos[0] + queen_pos[1]:    #on the positive diagonal

#         if obs_c > queen_pos[1]:
#             dr_max = min(dr_max, obs_c - queen_pos[1] - 1)
        
#         else:
#             ul_max = min(ul_max, queen_pos[1] - obs_c - 1)
    

#     elif obs_r - obs_c == queen_pos[0] - queen_pos[1]:  #on the negative diagonal

#         if obs_c > queen_pos[1]:
#             ur_max = min(ur_max, obs_c - queen_pos[1] - 1)
        
#         else:
#             dl_max = min(dl_max, queen_pos[1] - obs_c - 1)


#     elif obs_r == queen_pos[0]: #on the same row

#         if obs_c > queen_pos[1]:
#             r_max = min(r_max, obs_c - queen_pos[1] - 1)

#         else:
#             l_max = min(l_max, queen_pos[1] - obs_c - 1)


#     elif obs_c == queen_pos[1]: #on the same column

#         if obs_c > queen_pos[1]:
#             u_max = min(u_max, obs_r - queen_pos[0] - 1)
        
#         else:
#             d_max = min(d_max,queen_pos[0] - obs_r - 1 )


# print(ul_max + ur_max + dr_max + dl_max + r_max + l_max + u_max + d_max)




_ = "ACM ICPC Team"

# import itertools

# topics = ['10101', '11110', '00010']

# # teams = list(itertools.combinations(topics,2))

# # res = list()

# # for i in teams:

# #     res.append([ 1 if any((i[0][x] == '1', i[1][x] == '1')) else 0 for x in range(len(i[0])) ])

# # max_topics = max(list(map(sum,res)))
# # teams_w_max_topics = len([x for x in res if sum(x) == max_topics])

# # print([max_topics,teams_w_max_topics]) 

# # #My solution wasn't scalable enough for acquire the full score.


# _ = '------------------------------------------------------------------------------------'

# #Another guy's soluition


# def acmTeam(topic):

#     # create teams
#     teams = itertools.combinations(topic, 2)

#     # obtain team knowledge for each team
#     def pairwise_or(team):
#         return [int(tm0) or int(tm1) for tm0, tm1 in zip(team[0], team[1])]
    
#     team_knowledge = map(pairwise_or, teams)

#     # for each team knowledge, count the total topics
#     team_max_topics = list(map(lambda tk: tk.count(1), team_knowledge))
#     tmaxt = max(team_max_topics)

#     return [tmaxt, team_max_topics.count(tmaxt)]

# print(acmTeam(topics))




_ = "Organizing Container of Balls"

# l = [[0,2,1],[1,1,1],[2,0,0]]

# res1 = sorted([sum(x) for x in l])

# print(res1)


# # res2 = list()

# # for i in range(len(l)):
# #     temp = list()
# #     for j in range(len(l[0])):
# #         temp.append(l[j][i])
# #     res2.append(temp)

# res2 = sorted([ sum([ l[y][x] for y in range(len(l[0])) ])  for x in range(len(l)) ])

# print(res2)

# # This problem has, from my opinion, a bad redaction making it difficult to get the constrains and therefore the solution.
# # But, basically, the solution was to sum up all the balls for each type and also sum up all the balls in each container in the initial state, and this last sums will represent each container capacity
# # The problem is solved if there are equal amount of ball per type as it is for each container capacity, by just checking that condition the solution could be returned as a 'Possible' or 'Impossible' and that's it.