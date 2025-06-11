'''
CHALLENGES INDEX

17. Letter Combinations of a Phone Number (Hash Table) (BT)
22. Generate Parentheses (DP) (BT)
46. Permutations (Array) (BT)
78. Subsets (Array) (BT)
79. Word Search (Matrix) (BT)
131. Palindrome Partitioning (DP) (BT)
140. Word Break II (DP) (BT)
212. Word Search II (Array) (DFS) (BT) (Matrix)

39. Combination Sum (Array) (BT)
51. N-Queens (Matrix) (BT)
40. Combination Sum II (BT) (Array)


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


(11)
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
# #     '''

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

'''212. Word Search II'''
# def x():

#     # Input
#     # Case 1
#     board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
#     words = ["oath","pea","eat","rain"]
#     # Output: ["eat","oath"]

#     # Case 2
#     board = [["a","b"],["c","d"]], 
#     words = ["abcb"]
#     # Output: []

#     # Custom Case
#     board = [["a","b"],["c","d"]], 
#     words = ["abcb"]
#     # Output: []


#     '''
#     My Approach
    
#         Intuiton:

#             - Based on the 'Word Seach I' backtracking solution, I will try to emulate the same but
#                 since now there are multiple word to lookout for, I will rely on a Trie implementation
#                 to look out for prefixes to optimize the process.

#                 And to try to make it work, I will pull the first letter of each word and only start
#                 the searches from those positions, so, roughly the plan is:

#                 1. Collect the coordinates of the first letter from each of the word and store them in a dict
#                     as {'word': coordinates[(x,y)]}, if a word has no coordinates and it means it won't be found
#                     in the matrix, so it won't be in Trie.
                
#                 2. Initiate the Trie with the words with coordinates.

#                 3. Iterate through each of the words, and iterate for each pair of coordinates to look out for that word,
#                     if found, add it to a result list if don't pass to the next pair of coordinates, and so on for each word.
                
#                 4. Return the found words.
#     '''

#     'ACTUAL CODE'
#     # TRIE IMPLEMENTATION

#     # TrieNode Definition
#     class TrieNode:

#         def __init__(self):
#             self.values = {}
#             self.is_word = False


#     # Trie DS Definition
#     class Trie:

#         def __init__(self):
#             self.root = TrieNode()
        
#         def insert(self, word:str) -> None:

#             curr_node = self.root

#             for char in word:

#                 if char not in curr_node.values:
#                     curr_node.values[char] = TrieNode()
                
#                 curr_node = curr_node.values[char]
            
#             curr_node.is_word = True

#         def search(self, word:str) -> bool:

#             curr_node = self.root

#             for char in word:

#                 if char not in curr_node.values:
#                     return False
                
#                 curr_node = curr_node.values[char]

#             return curr_node.is_word

#         def stars_with(self, prefix:str) -> bool:

#             curr_node = self.root

#             for char in prefix:

#                 if char not in curr_node.values:
#                     return False
                
#                 curr_node = curr_node.values[char]

#             return True

#     'Actual Solution'
#     def findWords(board: list[list[str]], words: list[str]) -> list[str]:

#         import copy

#         #AUX BACKTRACK FUNC DEF
#         def backtrack(i:int, j:int, k:str) -> bool:

#             if new_trie.search(k):
#                 return True
                    
#             if not new_trie.stars_with(k):
#                 return False
            
#             temp = board[i][j]
#             board[i][j] = '.'

#             #1
#             if 0<i<len(board)-1 and 0<j<len(board[0])-1:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True
            
#             #2
#             elif 0 == i and 0 == j:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i, j+1, k+board[i][j+1]):        
#                     return True
                
#             #3
#             elif 0 == i and 0<j<len(board[0])-1:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i, j+1, k+board[i][j+1]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True
            
#             #4
#             elif len(board)-1 == i and len(board[0])-1 == j:
#                 if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True
            
#             #5
#             elif 0<i<len(board)-1 and 0 == j:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]):        
#                     return True
                
#             #6
#             elif 0<i<len(board)-1 and len(board[0])-1 == j:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True
            
#             #7
#             elif len(board)-1 == i and 0 == j:
#                 if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]):        
#                     return True
            
#             #8
#             elif len(board)-1 == i and 0<j<len(board[0])-1:
#                 if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True

#             #9
#             elif len(board)-1 == i and len(board[0])-1 == j:
#                 if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True


#             board[i][j] = temp

#             return False 
        

#         # COLLECT FIRST LETTER COORDINATES FOR EACH WORD
#         words_dict = {}

#         for word in words:

#             coordinates = []

#             for i,row in enumerate(board):
#                 coordinates.extend([(i,j) for j,elem in enumerate(row) if board[i][j] == word[0]])

#             if coordinates:
#                 words_dict[word] = coordinates


#         # INITIATE THE TRIE
#         new_trie = Trie()

#         for word in words_dict.keys():
#             new_trie.insert(word)

#         x = 0

#         result = []

#         # ITERATE THE DICT
#         for word in words_dict:

#             temp_board = copy.deepcopy(board)

#             for i,j in words_dict[word]:

#                 if backtrack(i, j, word[0]):

#                     result.append(word)
#                     board = temp_board

#         return result

#     # Testing
#     print(findWords(board=board, words=words))

#     '''
#     Notes:
#         My solution and approach wasn't that far. The logic was correct, the execution was the one to fail.
#         My version of the solution tends to get redundant and can't handle efficiently larger inputs
#     '''

#     # TrieNode Definition
#     class TrieNode:

#         def __init__(self):
#             self.values = {}
#             self.is_word = False

#     # Trie DS Definition
#     class Trie:

#         def __init__(self):
#             self.root = TrieNode()
        
#         def insert(self, word:str) -> None:

#             curr_node = self.root

#             for char in word:
#                 if char not in curr_node.values:
#                     curr_node.values[char] = TrieNode()            
#                 curr_node = curr_node.values[char]
            
#             curr_node.is_word = True

#     'Actual Solution'
#     def findWords(board: list[list[str]], words: list[str]) -> list[str]:

#         # Build the Trie
#         trie = Trie()

#         for word in words:
#             trie.insert(word)
        
#         # Auxiliary vars
#         rows, cols = len(board), len(board[0])
#         result = set()
#         visited = set()

#         #Aux DFS Func
#         def dfs(node:TrieNode, i:int, j:str, path:str) -> None:

#             if i<0 or i>=rows or j<0 or j>=cols or (i,j) in visited or board[i][j] not in node.values:
#                 return
            
#             visited.add((i,j))
#             node = node.values[board[i][j]]
#             path += board[i][j]

#             if node.is_word:
#                 result.add(path)
#                 node.is_word = False    # To avoid duplicate results

#             # Explore neighbors in 4 directions (up, down, left, right)
#             for x, y in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
#                 dfs(node, x, y, path)
            
#             visited.remove((i,j))        

#         # Traverse the board
#         for i in range(rows):
#             for j in range(cols):
#                 dfs(trie.root, i, j, '')        

#         return result

#     'Done'




'''39. Combination Sum'''
# def x():

#     from typing import List

#     # Input
#     # Case 1
#     candidates = [2,3,6,7]
#     target = 7
#     # Output: [[2,2,3],[7]]
    
#     # Case 2
#     candidates = [2,3,5]
#     target = 8
#     # Output: [[2,2,2,2],[2,3,3],[3,5]]
    
#     # Case 3
#     candidates = [2]
#     target = 1
#     # Output: []


#     '''
#     Explanation

#         Backtracking Strategy:

#             * Start with an empty combination.
#             * Explore each number in the candidates list, adding it to the combination.
#             * If the sum exceeds target, stop exploring further for that combination (backtrack).
#             * If the sum equals target, you've found a valid combination, so add it to the result.
#             * If the sum is still less than target, continue exploring by using the same number again (because we can reuse numbers).

#         Explanation:

#             1. Sorting: We sort the candidates array. This isn't strictly necessary but helps to optimize the solution. 
#                 By sorting, we can stop early when a number exceeds the remaining target.

#             2. Backtracking Function (backtrack):

#                 - remaining: This is the remaining sum we need to reach the target.
#                 - combination: The current combination of numbers we are considering.
#                 - start: This tells us from which index in candidates to start exploring. This is important to avoid using numbers that come before the current index
#                  (thus ensuring we don't generate duplicate combinations).
            
#             3. For loop: We loop through each candidate starting from the start index. If the current candidate is greater than the remaining sum, we stop 
#                 (because adding it would exceed the target).

#             4. Recursive Call: For each valid candidate, we add it to the combination and call backtrack recursively, reducing the target by the value of the candidate.

#             5. Backtracking Step: After returning from the recursive call, we remove the last added candidate from the combination to explore other possibilities 
#                 (this is the backtracking part).

#     '''

#     def combinationSum(candidates: List[int], target: int) -> List[List[int]]:

#         result = []

#         # Helper function to perform backtracking
#         def backtrack(remaining, combination, start):

#             # Base case: if remaining target is 0, add the combination to the result
#             if remaining == 0:
#                 result.append(list(combination))
#                 return
            
#             # Explore all candidates starting from 'start' index
#             for i in range(start, len(candidates)):

#                 # If the current candidate exceeds the remaining target, we stop
#                 if candidates[i] > remaining:
#                     break
                
#                 # Choose the current candidate and add it to the combination
#                 combination.append(candidates[i])
                
#                 # Recursively call backtrack with the reduced target, and the same index
#                 # (because we can reuse the same number)
#                 backtrack(remaining - candidates[i], combination, i)
                
#                 # Backtrack by removing the last added candidate from the combination
#                 combination.pop()

#         # Sort the candidates to help with early stopping
#         candidates.sort()
        
#         # Start backtracking with the entire list of candidates
#         backtrack(target, [], 0)

#         return result

#     # Testing
#     print(combinationSum(candidates=candidates, target=target))

#     '''Note: Done'''

'''51. N-Queens'''
# def x():

#     from typing import Optional

#     # Input
#     # Case 1
#     n = 4
#     # Output: [
#     #   [".Q..","...Q","Q...","..Q."],
#     #   ["..Q.","Q...","...Q",".Q.."]
#     # ]

#     # Case 2
#     n = 1
#     # Output: [Q]
    
#     # Case 2
#     n = 9
#     # Output: [Q]

#     '''
#     Explanation
        
#         The "N-Queens" problem is a classic example of a backtracking algorithm challenge. In this problem, you are asked to place N queens on an N x N chessboard 
#         such that no two queens attack each other. Queens can attack other queens that are in the same row, column, or diagonal.

        
#         Problem breakdown

#             1. You need to place N queens on an N x N board.
#             2. No two queens can be placed on the same row, column, or diagonal.
#             3. The goal is to return all possible arrangements that satisfy the above conditions.

        
#         Approach: Backtracking
            
#             We will solve this using backtracking, which is an algorithmic technique for solving constraint satisfaction problems. 
#             The idea is to place queens one by one and check for validity at each step. If a solution is found, we record it; otherwise, 
#             we backtrack and try different placements.
        
        
#         Steps:

#             1. Place Queens Row by Row: Start placing queens from the first row to the last. At each row, try placing a queen in each column, one by one, and check if it’s a valid position.

#             2. Check for Conflicts: After placing a queen, check whether it’s under attack from previously placed queens (i.e., check if there is a conflict in columns or diagonals).

#             3. Backtrack: If a conflict is found, remove the queen and try placing it in a different column. If no valid position exists in a row, backtrack to the previous row and move that queen.

#             4. Store Valid Solutions: When all queens are placed successfully, store the board configuration.
#     '''

#     def solveNQueens(n):

#         def is_valid(board, row, col):

#             # Check the column
#             for i in range(row):
#                 if board[i] == col:
#                     return False
                
#             # Check the diagonal (left-up)
#             for i, j in zip(range(row-1, -1, -1), range(col-1, -1, -1)):
#                 if board[i] == j:
#                     return False
                
#             # Check the diagonal (right-up)
#             for i, j in zip(range(row-1, -1, -1), range(col+1, n)):
#                 if board[i] == j:
#                     return False
                
#             return True


#         def backtrack(row, board, solutions):

#             if row == n:

#                 # If all queens are placed, convert the board to the required output format
#                 solutions.append(["." * col + "Q" + "." * (n - col - 1) for col in board])
#                 return
            
#             for col in range(n):

#                 if is_valid(board, row, col):
                    
#                     board[row] = col
#                     backtrack(row + 1, board, solutions)
#                     board[row] = -1  # Backtrack


#         # Initialize board and solutions list
#         solutions = []
#         board = [-1] * n  # Keeps track of queen's position in each row
#         backtrack(0, board, solutions)

#         return solutions


#     # Testing
#     solution = solveNQueens(n=n)
#     print(f'# of Solution: {len(solution)}')
#     for i in solution[1]:
#         print(i)

#     '''Note: Done'''

"""40. Combination Sum II"""
# def x():
    
#     # Input
#     # Case 1
#     candidates = [10,1,2,7,6,1,5]
#     target = 8
#     # Output: [[1,1,6],[1,2,5],[1,7],[2,6]]

#     # Case 2
#     candidates = [2,5,2,1,2]
#     target = 5
#     # Output: [[1,2,2],[5]]

#     '''
#     Solution

#         1. Sort the input list of candidate numbers.

#             This is important for two reasons:
#                 - It allows us to skip over duplicate values efficiently.
#                 - It enables early stopping (pruning) when the current candidate exceeds the remaining target.
                
#         2. Use backtracking to explore all valid combinations.

#             The goal is to build combinations of numbers that sum up to the target, without reusing the same index more than once per combination, and avoiding duplicate combinations in the output.

#         3. Define a recursive backtracking function that takes:
#             start: the current index in the candidates list.
#             path: the current list of numbers being built as a potential solution.
#             remaining: the value left to reach the target sum.

#         4. Base cases inside the backtracking function:
#             If remaining == 0: the current path is a valid combination, so it is added to the result list.
#             If remaining < 0: the current path exceeds the target, so we return early (prune the branch).

#         5. Loop through the candidates starting from the start index:
            
#             If the current index i is greater than start and the current number is the same as the previous one, we skip it to avoid duplicates at the same level of recursion.
#             If the current candidate is greater than remaining, we break the loop because all further candidates will also be too large (thanks to sorting).
            
#             Otherwise:
#                 Add the current candidate to the path.
#                 ○ Recurse with i + 1 as the new start index (move forward to avoid reusing the same number).
#                 ○ After recursion, remove the last number from the path to backtrack and explore other options.

#         6. Return the list of all valid combinations found during the recursive exploration.
#     '''

#     def combinationSum2(candidates: list[int], target: int) -> list[list[int]]:
        
#         # Sort to group duplicates and help with pruning
#         candidates.sort()
        
#         result = []

#         def backtrack(start: int, path: list[int], remaining: int):
#             if remaining == 0:
#                 result.append(path[:])
#                 return
#             if remaining < 0:
#                 return

#             for i in range(start, len(candidates)):
#                 # Skip duplicates at the same recursive level
#                 if i > start and candidates[i] == candidates[i - 1]:
#                     continue

#                 # If the current number is greater than the remaining target, break
#                 if candidates[i] > remaining:
#                     break

#                 # Include current number and recurse
#                 path.append(candidates[i])
#                 backtrack(i + 1, path, remaining - candidates[i])  # move to i+1, not i
#                 path.pop()  # backtrack

#         backtrack(0, [], target)

#         return result 

#     # Testing
#     print(combinationSum2(candidates=candidates, target=target))

#     '''Note: Done'''






