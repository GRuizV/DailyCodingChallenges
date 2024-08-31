'''
CHALLENGES INDEX

17. Letter Combinations of a Phone Number (Hash Table) (BT)
22. Generate Parentheses (DP) (BT)
46. Permutations (Array) (BT)
78. Subsets (Array) (BT)
79. Word Search (Matrix) (BT)


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
*BFS & DFS: Breadth-First Search & Depth-First Search
*Arrays, Hash Tables & Matrices
*Sorting
*Heaps, Stacks & Queues
*Graphs & Trees
*Others


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

'78. Subsets'
# def x():

#     # Input
#     # Case 1
#     nums = [1,2,3]
#     # Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]

#     # Case 2
#     nums = [0]
#     # Output: [[],[0]]

#     '''
#     My Approach

#         Intuition:
#             - Perhaps with itertool something might be done
#     '''

#     def subsets(nums:list[int]) -> list[list[int]]:

#         from itertools import combinations

#         result = []

#         for i in range(len(nums)+1):

#             result.extend(list(map(list, combinations(nums,i))))

#         return result

#     # Testing
#     print(subsets(nums=nums))

#     'Notes: It actually worked'


#     'Another more algorithmic Approach'
#     def subsets(nums: list[int]) -> list[list[int]]:

#         arr = [[]]

#         for j in nums:

#             temp = []

#             for i in arr: 

#                 temp.append(i+[j])
            
#             arr.extend(temp)
        
#         return arr 

#     # Testing
#     print(subsets(nums))

#     'Notes: The guy who came up with this is genius'

'79. Word Search'
# def x():

#     # Input
#     # Case 1
#     board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
#     word = 'ABCCED'
#     # Output: True

#     # Case 2
#     board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
#     word = 'SEE'
#     # Output: True

#     # Case 3
#     board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
#     word = 'ABCB'
#     # Output: False

#     '''
#     Intuition:

#         The problem can be solved by traversing the grid and performing a depth-first search (DFS) for each possible starting position. 
#         At each cell, we check if the current character matches the corresponding character of the word. 
#         If it does, we explore all four directions (up, down, left, right) recursively until we find the complete word or exhaust all possibilities.

#         Approach

#             1. Implement a recursive function backtrack that takes the current position (i, j) in the grid and the current index k of the word.
#             2. Base cases:
#                 - If k equals the length of the word, return True, indicating that the word has been found.
#                 - If the current position (i, j) is out of the grid boundaries or the character at (i, j) does not match the character at index k of the word, return False.
#             3. Mark the current cell as visited by changing its value or marking it as empty.
#             4. Recursively explore all four directions (up, down, left, right) by calling backtrack with updated positions (i+1, j), (i-1, j), (i, j+1), and (i, j-1).
#             5. If any recursive call returns True, indicating that the word has been found, return True.
#             6. If none of the recursive calls returns True, reset the current cell to its original value and return False.
#             7. Iterate through all cells in the grid and call the backtrack function for each cell. If any call returns True, return True, indicating that the word exists in the grid. Otherwise, return False.
            
#     '''

#     'Backtracking (Recursive) Approach'
#     def exist(board: list[list[str]], word: str) -> bool:

#         def backtrack(i, j, k):

#             if k == len(word):
#                 return True
            
#             if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[k]:
#                 return False
            
#             temp = board[i][j]
#             board[i][j] = ''
            
#             if backtrack(i+1, j, k+1) or backtrack(i-1, j, k+1) or backtrack(i, j+1, k+1) or backtrack(i, j-1, k+1):
#                 return True
            
#             board[i][j] = temp

#             return False

#         for i in range(len(board)):

#             for j in range(len(board[0])):

#                 if backtrack(i, j, 0):

#                     return True

#         return False
            
#     # Testing
#     print(exist(board, word))






