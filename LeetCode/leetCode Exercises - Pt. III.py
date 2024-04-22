'79. Word Search'

# Input

# # Case 1
# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = 'ABCCED'
# # Output: True


# # Case 2
# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = 'SEE'
# # Output: True


# # Case 3
# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = 'ABCB'
# # Output: False


'''
Intuition:

    The problem can be solved by traversing the grid and performing a depth-first search (DFS) for each possible starting position. 
    At each cell, we check if the current character matches the corresponding character of the word. 
    If it does, we explore all four directions (up, down, left, right) recursively until we find the complete word or exhaust all possibilities.

    Approach

        1. Implement a recursive function backtrack that takes the current position (i, j) in the grid and the current index k of the word.
        2. Base cases:
            - If k equals the length of the word, return True, indicating that the word has been found.
            - If the current position (i, j) is out of the grid boundaries or the character at (i, j) does not match the character at index k of the word, return False.
        3. Mark the current cell as visited by changing its value or marking it as empty.
        4. Recursively explore all four directions (up, down, left, right) by calling backtrack with updated positions (i+1, j), (i-1, j), (i, j+1), and (i, j-1).
        5. If any recursive call returns True, indicating that the word has been found, return True.
        6. If none of the recursive calls returns True, reset the current cell to its original value and return False.
        7. Iterate through all cells in the grid and call the backtrack function for each cell. If any call returns True, return True, indicating that the word exists in the grid. Otherwise, return False.
        
'''


# # Backtracking (Recursive) Approach
# def exist(board: list[list[str]], word: str) -> bool:


#     def backtrack(i, j, k):

#         if k == len(word):
#             return True
        
#         if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[k]:
#             return False
        
#         temp = board[i][j]
#         board[i][j] = ''
        
#         if backtrack(i+1, j, k+1) or backtrack(i-1, j, k+1) or backtrack(i, j+1, k+1) or backtrack(i, j-1, k+1):
#             return True
        
#         board[i][j] = temp

#         return False
        


#     for i in range(len(board)):

#         for j in range(len(board[0])):

#             if backtrack(i, j, 0):

#                 return True
            

#     return False
        

# print(exist(board, word))




'88. Merge Sorted Array'

# Input

# # Case 1
# nums1 = [1,2,3,0,0,0]
# m = 3
# nums2 = [2,5,6]
# n = 3
# # Output: [1,2,2,3,5,6]

# # Case 2
# nums1 = [1]
# m = 1
# nums2 = []
# n = 0
# # Output: [1]

# # Case 3
# nums1 = [0]
# m = 0
# nums2 = [1]
# n = 1
# # Output: [1]

# # Custom case
# nums1 = [0,2,0,0,0,0,0]
# m = 2
# nums2 = [-1,-1,2,5,6]
# n = 5
# # Output: [1]

# # Custom case
# nums1 = [-1,1,0,0,0,0,0,0]
# m = 2
# nums2 = [-1,0,1,1,2,3]
# n = 6
# # Output: [1]



# Solution

# def merge(nums1, m, nums2, n):

#     if m == 0:
#         for i in range(n):
#             nums1[i] = nums2[i]

#     elif n != 0:

#         m = n = 0

#         while n < len(nums2):

#             if nums2[n] < nums1[m]:

#                 nums1[:m], nums1[m+1:] = nums1[:m] + [nums2[n]], nums1[m:-1]

#                 n += 1
#                 m += 1
            
#             else:

#                 if all([x==0 for x in nums1[m:]]):
#                     nums1[m] = nums2[n]
#                     n += 1
                    
#                 m += 1


# merge(nums1,m,nums2,n)

# print(nums1)





'91. Decode Ways'  

# Input

# # Case 1:
# s = '12'
# # Output: 2

# # Case 2:
# s = '226'
# # Output: 3

# # Case 3:
# s = '06'
# # Output: 0

# # Custom Case:
# s = '112342126815'
# # Output: 11



# My apporach

# def fib(n):

#     res = [1,1]

#     for _ in range(n-1):
#         res.append(res[-2] + res[-1])
          
#     return res[1:]


# def numDecodings(s:str) -> int:

#     if s[0] == '0':
#         return 0
    
#     if len(s) == 1:
#         return 1

#     substrings = []
#     subs = ''

#     if s[0] in ['1', '2']:
#         subs += s[0]

#     for i in range(1, len(s)+1):

#         if i == len(s):
#             if subs != '':
#                 substrings.append(subs)

#         elif (s[i] in ['1', '2']) or (s[i-1] in ['1', '2'] and s[i] <= '6'):
#             subs += s[i]

#         else:
#             substrings.append(subs)
#             subs = ''

#     cap = len(max(substrings, key=len))
#     possibilities = fib(cap)

#     res = 0

#     for i in substrings:

#         if i in '10' or '20':
#             res += 1

#         else:
#             res += possibilities[len(i)-1] 
    
#     return res


# print(numDecodings(s))


'''
Notes: 
    This solution met 48% of expected results, there are a couple of cases I left unanalyzed.
    Nevertheless, the logic of fibonaccying the parsing numbers works, perhaps with more time
    a solution through this approach could work.

'''



# Dynamic Programming Approach

# def numDecodings(self, s):
    
#     dp = {len(s):1}

#     def backtrack(i):

#         if i in dp:
#             return dp[i]

#         if s[i]=='0':
#             return 0

#         if i==len(s)-1:
#             return 1

#         res = backtrack(i+1)

#         if int(s[i:i+2])<27:
#             res+=backtrack(i+2)
            
#         dp[i]=res

#         return res

#     return backtrack(0)





'xxx' 