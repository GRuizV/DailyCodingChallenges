'''
CHALLENGES INDEX

17. Letter Combinations of a Phone Number (Hash Table) (BT)
22. Generate Parentheses (DP) (BT)
46. Permutations (Array) (BT)


*LL: Linked-Lists
*BS: Binary Search
*DP: Dynamic Programming
*RC: Recursion
*TP: Two-pointers
*FCD: Floyd's cycle detection (Hare & Tortoise approach)
*PS: Preffix-sum
*SW: Sliding-Window
*MEM: Memoization
*GRE: Greedy
*DQ: Divide and Conquer
*BT: Backtracking


(XX)

'''


'17. Letter Combinations of a Phone Number'
# def x():

#     # # Input
#     # s = '23'


#     # # My Approach

#     '''
#     Rationale: Brute-forcing 
#         Iterating by the number of characters there are.
#     '''

#     def letterCombinations(digits):

#         dic = {
#             '2':['a', 'b', 'c'],
#             '3':['d', 'e', 'f'],
#             '4':['g', 'h', 'i'],
#             '5':['j', 'k', 'l'],
#             '6':['m', 'n', 'o'],
#             '7':['p', 'q', 'r', 's'],
#             '8':['t', 'u', 'v'],
#             '9':['w', 'x', 'y', 'z'],
#             }


#         lists = []

#         for i in digits:
#             lists.append(dic[i])

        
#         comb = []

#         if not digits:
#             return comb


#         if len(digits) == 4:
#             for i in range(len(lists[0])):
#                 for j in range(len(lists[1])):
#                     for k in range(len(lists[2])):
#                         for l in range(len(lists[3])):
#                             # comb.append(f'{lists[0][i]}{lists[1][j]}{lists[2][k]}{lists[3][l]}')
#                             comb.append(lists[0][i]+lists[1][j]+lists[2][k]+lists[3][l])
    
#         elif len(digits) == 3:
#             for i in range(len(lists[0])):
#                 for j in range(len(lists[1])):
#                     for k in range(len(lists[2])):
#                         comb.append(f'{lists[0][i]}{lists[1][j]}{lists[2][k]}')

#         elif len(digits) == 2:
#             for i in range(len(lists[0])):
#                 for j in range(len(lists[1])):
#                     comb.append(f'{lists[0][i]}{lists[1][j]}')

#         elif len(digits) == 1:
#             for i in range(len(lists[0])):
#                 comb.append(f'{lists[0][i]}')


#         return comb


#     print(letterCombinations(s))

#     'Notes: It works but it could be better'


#     # Recursive Approach

#     def letterCombinations(digits):

#         dic = {
#             '2':['a', 'b', 'c'],
#             '3':['d', 'e', 'f'],
#             '4':['g', 'h', 'i'],
#             '5':['j', 'k', 'l'],
#             '6':['m', 'n', 'o'],
#             '7':['p', 'q', 'r', 's'],
#             '8':['t', 'u', 'v'],
#             '9':['w', 'x', 'y', 'z'],
#             }

#         lists = []

#         for i in digits:
#             lists.append(dic[i])


#         def combine_sublists(lst):

#             if len(lst) == 1:
#                 return lst[0]
            
#             result = []

#             for item in lst[0]:
#                 for rest in combine_sublists(lst[1:]):
#                     result.append(item + rest)
            
#             return result
        
#         if not digits:
#             return []
        
#         else:
#             return combine_sublists(lists)

#     print(letterCombinations(s))

#     'Notes: Works pretty well'


#     # Itertools Approach

#     import itertools

#     def letterCombinations(digits):

#         dic = {
#             '2':['a', 'b', 'c'],
#             '3':['d', 'e', 'f'],
#             '4':['g', 'h', 'i'],
#             '5':['j', 'k', 'l'],
#             '6':['m', 'n', 'o'],
#             '7':['p', 'q', 'r', 's'],
#             '8':['t', 'u', 'v'],
#             '9':['w', 'x', 'y', 'z'],
#             }

#         lists = []

#         for i in digits:
#             lists.append(dic[i])

#         if not digits:
#             return []
        
#         else:
#             return [''.join(comb) for comb in itertools.product(*lists)]
        

#     print(letterCombinations(s))

'22. Generate Parentheses'
# def x():

#     # Input
#     n = 3   # Expected Output: ['((()))', '(()())', '(())()', '()(())', '()()()']

#     # My Approach
#     def generateParenthesis(n):
    
#         if n == 1:
#             return ['()']

#         result = []

#         for i in generateParenthesis(n-1):
#             result.append('('+ i +')')
#             result.append('()'+ i )
#             result.append(i + '()')


#         return sorted(set(result))

#     print(generateParenthesis(4))

#     '''
#     Note: My solution kind of work but it was missing one variation, apparently with DFS is solved.
#     '''
#     # DFS Approach
#     def generateParenthesis(n):
        
#         res = []

#         def dfs (left, right, s):

#             if len(s) == 2*n:
#                 res.append(s)
#                 return

#             if left < n:
#                 dfs(left+1, right, s + '(')

#             if right < left:
#                 dfs(left, right+1, s + ')')

#         dfs(0,0,'')

#         return res

#     # Testing
#     print(generateParenthesis(4))

'46. Permutations'
# def x():

#     # Input

#     # Case 1
#     nums = [1,2,3] # Exp. Out: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

#     # Case 2
#     nums = [0,1] # Exp. Out: [[0,1],[1,0]]

#     # Case 3
#     nums = [1] # Exp. Out: [[1]]


#     # Solution
#     def permute(nums: list[int]) -> list[list[int]]:
        
#         if len(nums) == 0:
#             return []
        
#         if len(nums) == 1:
#             return [nums]
        
#         l = []

#         for i in range(len(nums)):

#             num = nums[i]
#             rest = nums[:i] + nums[i+1:]

#             for p in permute(rest):
#                 l.append([num] + p)
            
#         return l

#     'Done'






