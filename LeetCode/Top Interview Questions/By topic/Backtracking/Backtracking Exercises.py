'''
CHALLENGES INDEX

17. Letter Combinations of a Phone Number (Hash Table) (BT)
22. Generate Parentheses (DP) (BT)
46. Permutations (Array) (BT)
78. Subsets (Array) (BT)
79. Word Search (Matrix) (BT)
131. Palindrome Partitioning (DP) (BT)
140. Word Break II (DP) (BT)



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

'''131. Palindrome Partitioning'''
# def x():

#     # Input
#     # Case 1
#     s = 'aab'
#     # Output: [["a","a","b"],["aa","b"]]

#     # Custom Case
#     s = 'aabcdededcbada'
#     # Output: [["a","a","b"],["aa","b"]]

#     # Custom Case
#     s = 'aabcdededcbada'
#     # Output: ['abcdededcba', 'bcdededcb', 'cdededc', 'deded', 'ded', 'ede', 'ded', 'ada', 'aa'] 


#     '''
#     My Approach

#         Intuition:

#             Here I don't actually have much ideas in how to solve it, but one good approach
#             I think woul dbe to make a function that can pull all the palindroms present in a string.

#             that could be a good start point.
#     '''

#     def palindromes(string:str) -> list[str]:

#         s_len = len(string)
#         palindromes = []

#         for i in range(s_len, 1, -1):   # from s_len down to length 2 of substring
        
#             j = 0

#             while j + i <= s_len: 

#                 subs = string[j:j+i]

#                 if subs == subs[::-1]:

#                     palindromes.append(subs)

#                 j += 1

#         print(palindromes)

#     # Testing
#     # Printout: ['abcdededcba', 'bcdededcb', 'cdededc', 'deded', 'ded', 'ede', 'ded', 'ada', 'aa'] 
#     palindromes(string=s)

#     '''
#     Notes: At least this I was able to do, from here on, I am feeling I am going to brute forcing this and it won't end up being efficient.

#         I didn't actually solved it but I don't want to waste more time over this.
#     '''

'''140. Word Break II'''
# def x():

#     #Input
#     #Case 1
#     s = "catsanddog"
#     wordDict = ["cat","cats","and","sand","dog"]
#     #Output: ["cats and dog","cat sand dog"]

#     #Case 2
#     s = "pineapplepenapple"
#     wordDict = ["apple","pen","applepen","pine","pineapple"]
#     #Output: ["pine apple pen apple","pineapple pen apple","pine applepen apple"]

#     #Case 3
#     s = "catsandog"
#     wordDict = ["cats","dog","sand","and","cat"]
#     #Output: []


#     '''
#     My Approach

#         Intuition:

#             - With the solution of the last exercise, bring the found words into a list and join them to from a sentence.
#             - In a loop, check if the first found word is the same of the last sentece, if do, keep searching for another word,
#                 - if not found words after looping from the first character, end the loop.
#     '''

#     def wordBreak(s:str, wordDict:list[str]) -> list[str]:

#         sentences = []
#         sent = []
#         string = s
#         lasts_first_word = []

#         while True:

#             j = 0

#             while j < len(string):

#                 if string[0:j+1] in wordDict and string[0:j+1] not in lasts_first_word:

#                     sent.append(string[0:j+1])
#                     string = string[j+1:]
#                     j = 0
                
#                 else:
#                     j += 1
            

#             if sent:
#                 sentences.append(' '.join(sent))
#                 string = s
#                 lasts_first_word.append(sent[0])
#                 sent = []
            
#             else:
#                 break
        
#         return sentences        

#     # Testing
#     print(wordBreak(s=s, wordDict=wordDict))

#     "Note:This solution doesn't even get to pass all the initial test cases, but at least it worked as a challenge to do at least one"


#     'Backtracking & Recursion approach'
#     def wordBreakHelper(s:str, start:int, word_set:set, memo:dict) -> list[str]:

#         if start in memo:
#             return memo[start]
        
#         valid_substr = []

#         if start == len(s):
#             valid_substr.append('')

#         for end in range(start+1, len(s)+1):

#             prefix = s[start:end]

#             if prefix in word_set:

#                 suffixes = wordBreakHelper(s, end, word_set, memo)

#                 for suffix in suffixes:

#                     valid_substr.append(prefix + ('' if suffix == '' else ' ') + suffix)

#         memo[start] = valid_substr

#         return valid_substr
            

#     def wordBreak(s:str, wordDict: list[str]) -> list[str]:

#         memo = {}
#         word_set = set(wordDict)
#         return wordBreakHelper(s, 0, word_set, memo)

#     # Testing
#     print(wordBreak(s=s, wordDict=wordDict))

#     'Done'






