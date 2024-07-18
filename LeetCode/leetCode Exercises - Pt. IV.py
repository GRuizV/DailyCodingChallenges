'''
CHALLENGES INDEX

179. Largest Number
189. Rotate Array (TP)
198. House Robber (DS)
200. Number of Islands (Matrix) (BFS) (DFS)
202. Happy Number (FCD) (TP)
204. Count Primes
206. Reverse Linked List
207. Course Schedule (DFS)
210. Course Schedule II (DFS)
208. Implement Trie (Prefix Tree)
212. Word Search II (DFS)
215. Kth Largest Element in an Array (Heaps)
218. The Skyline Problem (Heaps)
227. Basic Calculator II (Stack)
230. Kth Smallest Element in a BST (RC) (Heaps) or (Stack)
234. Palindrome Linked List - Opt: (RC) + (TP) or (TP)
237. Delete Node in a Linked List
238. Product of Array Except Self (PS)
239. Sliding Window Maximum
240. Search a 2D Matrix II
279. Perfect Squares (DP)
283. Move Zeroes (TP)
287. Find the Duplicate Number (FCD)




*DS: Dynamic Programming
*RC: Recursion
*TP: Two-pointers
*FCD: Floyd's cycle detection
*PS: Preffix-sum
*SW: Sliding-Window
*FCD: Floyd's Cycle Detection (Hare & Tortoise)


(22)
'''




'''179. Largest Number'''

# Input

# # Case 1
# nums = [20,1]
# # Output: "201"

# # Case 2
# nums = [3,30,34,5,9]
# # Output: "9534330"

# # Custom Case
# nums = [8308,8308,830]
# # Output: "83088308830"



# My 1st Approach

# def largestNumber(nums: list[int]) -> str: 

#     nums = [str(x) for x in nums]
   
#     res = sorted(nums, key= lambda x: int(x[0]), reverse=True)  


#     # Mergesort
#     def mergesort(seq: list) -> list:

#         if len(seq) <= 1:
#             return seq

#         mid = len(seq)//2

#         left_side, right_side = seq[:mid], seq[mid:]

#         left_side = mergesort(left_side)
#         right_side = mergesort(right_side)

#         return merge(left=left_side, right=right_side)

#     # Auxiliary merge for Mergesort
#     def merge(left: list, right: list) -> list:

#         res = []
#         zeros = []
#         i = j = 0

#         while i < len(left) and j < len(right):

#             if left[i][-1] == '0':
#                 zeros.append(left[i])
#                 i+=1

#             elif right[j][-1] == '0':
#                 zeros.append(right[j])
#                 j+=1
            
#             elif left[i][0] == right[j][0]:

#                 if left[i]+right[j] > right[j]+left[i]:
#                     res.append(left[i])
#                     i+=1

#                 else:
#                     res.append(right[j])
#                     j+=1                

#             elif int(left[i][0]) > int(right[j][0]):
#                 res.append(left[i])
#                 i+=1
            
#             else:
#                 res.append(right[j])
#                 j+=1
        

#         while i < len(left):
#             res.append(left[i])
#             i+=1

        
#         while j < len(right):
#             res.append(right[j])
#             j+=1


#         # Deal with the elements with '0' as last digit
#         zeros.sort(key=lambda x: int(x), reverse=True)

#         return res+zeros          

#     result = mergesort(seq=res)
    
#     return ''.join(result)


# print(largestNumber(nums=nums))

'This approach cleared 57% of cases '


# My 2nd Approach

# def largestNumber(nums: list[int]) -> str: 

#     res = [str(x) for x in nums]
   
#     # res = sorted(nums, key= lambda x: int(x[0]), reverse=True)  


#     # Mergesort
#     def mergesort(seq: list) -> list:

#         if len(seq) <= 1:
#             return seq

#         mid = len(seq)//2

#         left_side, right_side = seq[:mid], seq[mid:]

#         left_side = mergesort(left_side)
#         right_side = mergesort(right_side)

#         return merge(left=left_side, right=right_side)

#     # Auxiliary merge for Mergesort
#     def merge(left: list, right: list) -> list:

#         res = []        
#         i = j = 0

#         while i < len(left) and j < len(right):

#             if left[i]+right[j] > right[j]+left[i]:
#                 res.append(left[i])
#                 i += 1

#             else:
#                 res.append(right[j])
#                 j += 1
        
#         while i < len(left):
#             res.append(left[i])
#             i += 1
                        
#         while j < len(right):
#             res.append(right[j])
#             j += 1

#         return res        

#     result = mergesort(seq=res)
    
#     return ''.join(result)


# print(largestNumber(nums=nums))

'This one did it!'




'''189. Rotate Array'''

# Input

# # Case 1
# nums, k = [1,2,3,4,5,6,7], 3
# # Output: [5,6,7,1,2,3,4]

# # Case 2
# nums, k = [-1,-100,3,99], 2
# # Output: [3,99,-1,-100]

# # My approach
# def rotate(nums: list[int], k: int) -> None:

#     if len(nums) == 1:
#         return
    
#     rot = k % len(nums)

#     dic = {k:v for k, v in enumerate(nums)}

#     for i in range(len(nums)):

#         n_idx = (i+rot)%len(nums)
#         nums[n_idx] = dic[i]

'It actually worked!'




'''198. House Robber'''
   
# Input

# # Case 1
# nums = [1,2,3,1]
# # Output: 4

# # Case 2
# nums = [2,7,9,3,1]
# # Output: 12

# # Custom Case
# nums = [2,1,1,2]
# # Output: 12

# # DS Approach ( space: O(n) )
# def rob(nums: list[int]) -> int:
    
#     # Handling corner cases
#     if len(nums) == 1:
#         return nums[0]
    
#     # Initializing the aux array
#     dp = [0] * len(nums)
#     dp[0] = nums[0]
#     dp[1] = max(dp[0], nums[1])

#     for i in range(2, len(nums)):

#         dp[i] = max(dp[i-1], dp[i-2] + nums[i])

#     return dp[-1]

# print(rob(nums=nums))
                
'-------------------'

# # DS Approach ( space: O(1) )
# def rob(nums: list[int]) -> int:
    
#     # Handling corner cases
#     if len(nums) == 1:
#         return nums[0]
    
#     # Initializing the aux array
#     prev_rob = 0
#     max_rob = 0

#     for num in nums:

#         temp = max(max_rob, prev_rob + num)
#         prev_rob = max_rob
#         max_rob = temp
    
#     return max_rob

# print(rob(nums=nums))

'Done'




'''200. Number of Islands'''

# Input

# # Case 1
# grid = [
#   ["1","1","1","1","0"],
#   ["1","1","0","1","0"],
#   ["1","1","0","0","0"],
#   ["0","0","0","0","0"]
# ]
# # Ouput: 1

# # Case 2
# grid = [
#   ["1","1","0","0","0"],
#   ["1","1","0","0","0"],
#   ["0","0","1","0","0"],
#   ["0","0","0","1","1"]
# ]
# # Ouput: 3

# # Custom Case
# grid = [
#     ["1","0"]
#     ]
# # Ouput: 1


'My BFS Approach'
# def numIslands(grid:list[list[str]]) -> int:
    
#     if len(grid) == 1:
#         return len([x for x in grid[0] if x =='1'])

#     # Create the 'lands' coordinates
#     coord = []

#     # Collecting the 'lands' coordinates
#     for i, row in enumerate(grid):
#         coord.extend((i, j) for j, value in enumerate(row) if value == '1')


#     # Create the groups holder
#     islands = []
#     used = set()


#     # BFS Definition
#     def bfs(root:tuple) -> list:

#         queue = [root]
#         curr_island = []

#         while queue:

#             land = queue.pop(0)
#             x, y = land[0], land[1]
            
#             if grid[x][y] == '1' and (land not in curr_island and land not in used):

#                 curr_island.append(land)
              
#                 # Define next lands to search
#                 if x == 0:
#                     if y == 0:
#                         next_lands = [(x+1,y),(x,y+1)]
                    
#                     elif y < len(grid[0])-1:
#                         next_lands = [(x+1,y),(x,y-1),(x,y+1)]
                    
#                     else:
#                         next_lands = [(x+1,y),(x,y-1)]
                
#                 elif x < len(grid)-1:
#                     if y == 0:
#                         next_lands = [(x-1,y),(x+1,y),(x,y+1)]
                    
#                     elif y < len(grid[0])-1:
#                         next_lands = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
                    
#                     else:
#                         next_lands = [(x-1,y),(x+1,y),(x,y-1)]
                
#                 else:
#                     if y == 0:
#                         next_lands = [(x-1,y),(x,y+1)]
                    
#                     elif y < len(grid[0])-1:
#                         next_lands = [(x-1,y),(x,y-1),(x,y+1)]
                    
#                     else:
#                         next_lands = [(x-1,y),(x,y-1)]
                                   
#                 # List the next lands to visit
#                 for next_land in next_lands:

#                     if next_land not in curr_island:

#                         queue.append(next_land)

#         return curr_island
        

#     # Checking all the 1s in the grid
#     for elem in coord:

#         if elem not in used:

#             island = bfs(elem)

#             islands.append(island)
#             used.update(set(island))
    
#     return len(islands)


# print(numIslands(grid=grid))


'Simplified & Corrected BFS Approach'

# def numIslands(grid:list[list[str]]) -> int:

#     if not grid:
#         return 0

#     num_islands = 0
#     directions = [(1,0),(-1,0),(0,1),(0,-1)]

#     for i in range(len(grid)):

#         for j in range(len(grid[0])):

#             if grid[i][j] == '1':

#                 num_islands += 1

#                 queue = [(i,j)]

#                 while queue:

#                     x, y = queue.pop(0)

#                     if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == '1':

#                         grid[x][y] = '0'    # Mark as visited

#                         for dx, dy in directions:

#                             queue.append((x + dx, y + dy))
    
#     return num_islands

'Done'




'''202. Happy Number'''

# Input

# # Case 1
# n = 19
# # Output: True

# # Case 2
# n = 2
# # Output: False

# # Custom Case
# n = 18
# # Output: False


# My Approach
'''
Intuition (Recursive)
    
    - Recursively separate the digits and check the sum of their squares compared to 1.
        - If the stackoverflow is reached, return False
    
    '''

# def isHappy(n:int) -> bool:

#     def aux(m:int) -> bool:

#         num = [int(x)**2 for x in str(m)]
#         num = sum(num)

#         if num == 1:
#             return True
        
#         return aux(m=num)
    
#     try:
#         res = aux(m=n)

#         if res:
#             return True
    
#     except RecursionError as e:        
#         return False

# print(isHappy(n=n))

'This approach may work but it exceed time limit: only met 4% of cases'


'''
Notes: 

There are mainly two ways of solving this: The set approach and the Floyd's Cycle detection algorithm

    - The set approach: Use a set to save the seen numbers and if you end up in one of them, you entered a cycle
    - The Floyd's Cycle Detection Algorithm: Similar to the case of catching a cycle in a linked list with two pointers: Slow and Fast.
'''

# # Set Approach
# def isHappy(n:int) -> bool:

#     def getNum(m:int)->int:
#         return sum(int(x)**2 for x in str(m))

#     seen = set()

#     while n != 1 and n not in seen:
#         seen.add(n)
#         n = getNum(n)
    
#     return n == 1

# print(isHappy(n=n))


# # # FDC Approach
# def isHappy(n:int) -> bool:

#     def getNum(m:int)->int:
#         return sum(int(x)**2 for x in str(m))

#     slow = n
#     fast = getNum(n)

#     while fast != 1 and slow != fast:
#         slow = getNum(slow)
#         fast = getNum(getNum(fast))
    
#     return fast == 1

# print(isHappy(n=n))
'Done'




'''204. Count Primes'''

# Input

# # Case 1
# n = 10
# # Output: 4 (2,3,5,7)

# # Custom Case
# n = 30
# # Output: 4 (2,3,5,7)


'''
Intuition
    - Application of Eratosthenes Sieve
'''

# def countPrimes(n: int) -> int:

#     # Handling corner cases
#     if n in range(3):
#         return 0 
    
        
#     primes, non_primes = [], []

#     for num in range(2, n):

#         primes.append(num) if num not in non_primes else None

#         non_primes.extend(x for x in range(num*num, n, num))
    
#     return len(primes)

# print(countPrimes(n=n))

'''
This solution works well for data input in low scales (Worked for 26% of the cases), for big numbers could be quite time complex.

After researching a modified version of the Sieve is the way to go, instead of appending numbers to later count them, creating a boolean list to only mark
the multiples of other primes is more time and space efficient than storing the actual numbers.

    But the real hit here is that we will curb the loop of marking the multiples to the square root of the parameter given, because is safe to assume that after the square root
    other numbers will pretty much be multiples of the range before the SR.

'''

# def countPrimes(n:int) -> int:

#     if n <= 2:
#         return 0

#     primes = [True]*n

#     primes[0] = primes[1] = False

#     for i in range(2, int(n**0.5)+1):

#         if primes[i]:

#             for j in range(i*i, n, i):
#                 primes[j] = False
    
#     return sum(primes)

'This one did it!'




'''206. Reverse Linked List'''

# Base 

# # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next



# Input

# # Case 1
# head_layout = [1,2,3,4,5]
# head = ListNode(val=1)
# two, three, four, five = ListNode(2), ListNode(3), ListNode(4), ListNode(5),
# head.next, two.next, three.next, four.next = two, three, four, five
# # Output: [5,4,3,2,1]

# # Case 2
# head_layout = [1,2]
# head, two, = ListNode(1), ListNode(2)
# head.next = two
# # Output: [2,1]

# # Case 3
# head_layout = []
# head = None
# # Output: []


# My Approach

# def reverseList(head:ListNode) -> ListNode:
    
#     # Initialize node holders
#     prev = None
#     curr = head    

#     while curr:
#         next_node = curr.next
#         curr.next = prev
#         prev = curr
#         curr = next_node       
    
#     return prev


# def rec_reverseList(head:ListNode) -> ListNode:
    
#     # Base case
#     if not head or not head.next:
#         return head   

#     # Recursive Call
#     new_head = rec_reverseList(head.next)

#     # Reversing the list
#     head.next.next = head
#     head.next = None

#     return new_head

'Done'




'''207. Course Schedule'''

# Input

# # Case 1
# numCourses = 2
# prerequisites = [[1,0]]
# # Output: True

# # Case 2
# numCurses = 2
# prerequisites = [[1,0], [0,1]]
# # Output: False


# # DFS Approach

# def canFinish(numCourses:int, prerequisites: list[list[int]]) -> bool:

#     # Create the graph
#     preMap = {course:[] for course in range(numCourses)}

#     # Populate the graph
#     for crs, pre in prerequisites:
#         preMap[crs].append(pre)

#     # Create a visit (set) to check the current branch visited (to detect cycles)
#     visit_set = set()

#     # Define the DFS func
#     def dfs(node):

#         # Base case where is a cylce
#         if node in visit_set:
#             return False
        
#         # Base case where not prerequisites
#         if preMap[node] == []:
#             return True
        
#         visit_set.add(node)

#         for prereq in preMap[node]:
            
#             if not dfs(prereq):
#                 return False

#         visit_set.remove(node)
#         preMap[prereq] = [] # As it passes, then cleared the list in case is a prereq of something else
#         return True
    
#     courses = sorted(set(x for pair in prerequisites for x in pair))

#     for crs in courses:        
#         if not dfs(crs):
#             return False
    
#     return True


# print(canFinish(numCourses, prerequisites))

'Done'




'''210. Course Schedule II'''

# Input

# # Case 1
# numCourses = 2
# prerequisites = [[0,1]]
# # Output: True

# # Case 2
# numCourses = 4
# prerequisites = [[1,0],[2,0],[3,1],[3,2]]
# # Output: [0,1,2,3] or [0,2,1,3]

# # Case 3
# numCourses = 1
# prerequisites = []
# # Output: [0]

# # Custom Case
# numCourses = 3
# prerequisites = [[1,0]]
# # Output: [0]


# My approach

# def findOrder(numCourses:int, prerequisites: list[list[int]]) -> list[int]:

#     # Handling corner case
#     if not prerequisites:
#         return [x for x in range(numCourses)]
    
#     # Create the graph as an Adjacency list
#     pre_map = {course:[] for course in range(numCourses)}

#     # Populate the graph
#     for crs, pre in prerequisites:
#         pre_map[crs].append(pre)

#     # Create the visit set to watch for cycles
#     visit_set = set()

#     # Create the path in which the order of the courses will be stored
#     path = []

#     # Define the recursive dfs func
#     def dfs(course):

#         # If we get to a course we already pass through, means we're in a Cycle
#         if course in visit_set:
#             return False

#         # If we get to a course that has no prerequisites, means we can take it
#         if pre_map[course] == []:

#             path.append(course) if course not in path else None

#             return True
        
#         visit_set.add(course)   # Mark the course as visited

#         # Check if the course's prerequisites are available to take
#         for prereq in pre_map[course]:
            
#             if dfs(prereq) is False:
#                 return False
            
#         visit_set.remove(course)
#         pre_map[course] = []
#         path.append(course)  # Build the path backwards

#         return True


#     # # Create a list with all the courses available
#     # courses = sorted(set(x for pair in prerequisites for x in pair))


#     # Run through all the courses
#     for crs in range(numCourses):
#         if dfs(crs) is False:
#             return []
        
#     return path

# print(findOrder(numCourses=numCourses, prerequisites=prerequisites))

'It worked based on the first case version'




'''208. Implement Trie (Prefix Tree)'''

# # Implementation

# class TrieNode:

#     def __init__(self, is_word=False):

#         self.values = {}
#         self.is_word = is_word


# class Trie:

#     def __init__(self):
#         self.root = TrieNode()
   

#     def insert(self, word: str) -> None:

#         node = self.root

#         for char in word:

#             if char not in node.values:
#                 node.values[char] = TrieNode()
            
#             node = node.values[char]

#         node.is_word = True


#     def search(self, word: str) -> bool:
        
#         node = self.root

#         for char in word:          
                    
#             if char not in node.values:
#                 return False
            
#             node = node.values[char]
        
#         return node.is_word


#     def startsWith(self, prefix: str) -> bool:
        
#         node = self.root

#         for char in prefix:

#             if char not in node.values:
#                 return False
            
#             node = node.values[char]
        
#         return True

# # Trie Testing
# new_trie = Trie()

# new_trie.insert('Carrot')

# print(new_trie.startsWith('Car'))
# x=''

'Done'




'''212. Word Search II'''

# Input 

# # Case 1
# board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
# words = ["oath","pea","eat","rain"]
# # Output: ["eat","oath"]

# # Case 2
# board = [["a","b"],["c","d"]], 
# words = ["abcb"]
# # Output: []

# # Custom Case
# board = [["a","b"],["c","d"]], 
# words = ["abcb"]
# # Output: []


# My Approach

'''
Rationale:

    - Based on the 'Word Seach I' backtracking solution, I will try to emulate the same but
        since now there are multiple word to lookout for, I will rely on a Trie implementation
        to look out for prefixes to optimize the process.

        And to try to make it work, I will pull the first letter of each word and only start
        the searches from those positions, so, roughly the plan is:

        1. Collect the coordinates of the first letter from each of the word and store them in a dict
            as {'word': coordinates[(x,y)]}, if a word has no coordinates and it means it won't be found
            in the matrix, so it won't be in Trie.
        
        2. Initiate the Trie with the words with coordinates.

        3. Iterate through each of the words, and iterate for each pair of coordinates to look out for that word,
            if found, add it to a result list if don't pass to the next pair of coordinates, and so on for each word.
        
        4. Return the found words

'''

# # ACTUAL CODE
# # TRIE IMPLEMENTATION

# # TrieNode Definition
# class TrieNode:

#     def __init__(self):
#         self.values = {}
#         self.is_word = False


# # Trie DS Definition
# class Trie:

#     def __init__(self):
#         self.root = TrieNode()
    
#     def insert(self, word:str) -> None:

#         curr_node = self.root

#         for char in word:

#             if char not in curr_node.values:
#                 curr_node.values[char] = TrieNode()
            
#             curr_node = curr_node.values[char]
        
#         curr_node.is_word = True

#     def search(self, word:str) -> bool:

#         curr_node = self.root

#         for char in word:

#             if char not in curr_node.values:
#                 return False
            
#             curr_node = curr_node.values[char]

#         return curr_node.is_word

#     def stars_with(self, prefix:str) -> bool:

#         curr_node = self.root

#         for char in prefix:

#             if char not in curr_node.values:
#                 return False
            
#             curr_node = curr_node.values[char]

#         return True

# # Actual Solution
# def findWords(board: list[list[str]], words: list[str]) -> list[str]:

#     import copy

#     #AUX BACKTRACK FUNC DEF
#     def backtrack(i:int, j:int, k:str) -> bool:

#         if new_trie.search(k):
#             return True
                 
#         if not new_trie.stars_with(k):
#             return False
        
#         temp = board[i][j]
#         board[i][j] = '.'

#         #1
#         if 0<i<len(board)-1 and 0<j<len(board[0])-1:
#             if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]) or backtrack(i, j-1, k+board[i][j-1]):        
#                 return True
        
#         #2
#         elif 0 == i and 0 == j:
#             if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i, j+1, k+board[i][j+1]):        
#                 return True
            
#         #3
#         elif 0 == i and 0<j<len(board[0])-1:
#             if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i, j+1, k+board[i][j+1]) or backtrack(i, j-1, k+board[i][j-1]):        
#                 return True
        
#         #4
#         elif len(board)-1 == i and len(board[0])-1 == j:
#             if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j-1, k+board[i][j-1]):        
#                 return True
        
#         #5
#         elif 0<i<len(board)-1 and 0 == j:
#             if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]):        
#                 return True
            
#         #6
#         elif 0<i<len(board)-1 and len(board[0])-1 == j:
#             if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j-1, k+board[i][j-1]):        
#                 return True
        
#         #7
#         elif len(board)-1 == i and 0 == j:
#             if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]):        
#                 return True
        
#         #8
#         elif len(board)-1 == i and 0<j<len(board[0])-1:
#             if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]) or backtrack(i, j-1, k+board[i][j-1]):        
#                 return True

#         #9
#         elif len(board)-1 == i and len(board[0])-1 == j:
#             if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j-1, k+board[i][j-1]):        
#                 return True


#         board[i][j] = temp

#         return False 
    

#     # COLLECT FIRST LETTER COORDINATES FOR EACH WORD
#     words_dict = {}

#     for word in words:

#         coordinates = []

#         for i,row in enumerate(board):
#             coordinates.extend([(i,j) for j,elem in enumerate(row) if board[i][j] == word[0]])

#         if coordinates:
#             words_dict[word] = coordinates


#     # INITIATE THE TRIE
#     new_trie = Trie()

#     for word in words_dict.keys():
#         new_trie.insert(word)

#     x = 0

#     result = []

#     # ITERATE THE DICT
#     for word in words_dict:

#         temp_board = copy.deepcopy(board)

#         for i,j in words_dict[word]:

#             if backtrack(i, j, word[0]):

#                 result.append(word)
#                 board = temp_board
            
#     x = 0

#     return result

# print(findWords(board=board, words=words))

'''
Notes:
    My solution and approach wasn't that far. The logic was correct, the execution was the one to fail.
    My version of the solution tends to get redundant and can't handle efficiently larger inputs
'''


# # TrieNode Definition
# class TrieNode:

#     def __init__(self):
#         self.values = {}
#         self.is_word = False


# # Trie DS Definition
# class Trie:

#     def __init__(self):
#         self.root = TrieNode()
    
#     def insert(self, word:str) -> None:

#         curr_node = self.root

#         for char in word:
#             if char not in curr_node.values:
#                 curr_node.values[char] = TrieNode()            
#             curr_node = curr_node.values[char]
        
#         curr_node.is_word = True


# # Actual Solution
# def findWords(board: list[list[str]], words: list[str]) -> list[str]:

#     # Build the Trie
#     trie = Trie()

#     for word in words:
#         trie.insert(word)
    
#     # Auxiliary vars
#     rows, cols = len(board), len(board[0])
#     result = set()
#     visited = set()


#     #Aux DFS Func
#     def dfs(node:TrieNode, i:int, j:str, path:str) -> None:

#         if i<0 or i>=rows or j<0 or j>=cols or (i,j) in visited or board[i][j] not in node.values:
#             return
        
#         visited.add((i,j))
#         node = node.values[board[i][j]]
#         path += board[i][j]

#         if node.is_word:
#             result.add(path)
#             node.is_word = False    # To avoid duplicate results

#         # Explore neighbors in 4 directions (up, down, left, right)
#         for x, y in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
#             dfs(node, x, y, path)
        
#         visited.remove((i,j))
       

#     # Traverse the board
#     for i in range(rows):
#         for j in range(cols):
#             dfs(trie.root, i, j, '')        


#     return result

'Done'




'''215. Kth Largest Element in an Array'''

# import heapq

# def findKthLargest(self, nums: list[int], k: int) -> int:
#         heap = nums[:k]
#         heapq.heapify(heap)
        
#         for num in nums[k:]:
#             if num > heap[0]:
#                 heapq.heappop(heap)
#                 heapq.heappush(heap, num)
        
#         return heap[0]

'Done'




'''218. The Skyline Problem'''

'''
Explanation of the Code

    Events Creation:

        For each building, two events are created: entering ((left, -height, right)) and exiting ((right, height, 0)).
    
    Sorting Events:

        Events are sorted first by x-coordinate. If x-coordinates are the same, entering events are processed before exiting events. For entering events with the same x-coordinate, taller buildings are processed first.
    
    Processing Events:

        A max-heap (live_heap) keeps track of the current active buildings' heights. Heights are stored as negative values to use Python's min-heap as a max-heap.
        When processing each event, heights are added to or removed from the heap as needed.
        If the maximum height changes (top of the heap), a key point is added to the result.
    
    This approach efficiently manages the skyline problem by leveraging sorting and a max-heap to dynamically track the highest building at each critical point.
'''

# from heapq import heappush, heappop, heapify

# def getSkyline(buildings: list[list[int]]) -> list[list[int]]:
        
#     # Create events for entering and exiting each building
#     events = []

#     for left, right, height in buildings:
#         events.append((left, -height, right))  # Entering event
#         events.append((right, height, 0))     # Exiting event
    

#     # Sort events: primarily by x coordinate, then by height
#     events.sort()
    

#     # Max-heap to store the current active buildings
#     result = []
#     live_heap = [(0, float('inf'))]  # (height, end)


#     # Process each event
#     for x, h, r in events:

#         if h < 0:  # Entering event
#             heappush(live_heap, (h, r))

#         else:  # Exiting event
            
#             # Remove the building height from the heap
#             for i in range(len(live_heap)):
#                 if live_heap[i][1] == x:
#                     live_heap[i] = live_heap[-1]  # Replace with last element
#                     live_heap.pop()  # Remove last element
#                     heapify(live_heap)  # Restore heap property
#                     break
        
#         # Ensure the heap is valid
#         while live_heap[0][1] <= x:
#             heappop(live_heap)
        
#         # Get the current maximum height
#         max_height = -live_heap[0][0]
        
#         # If the current maximum height changes, add the key point
#         if not result or result[-1][1] != max_height:
#             result.append([x, max_height])
                
#     return result

'Done'




'''227. Basic Calculator II'''

# Input

# # Case 1
# s = "3+2*2"
# # Output: 7

# # Case 2
# s = " 3/2 "
# # Output: 1

# # Case 3
# s = " 3+5 / 2 "
# # Output: 5

# # Custom Case
# s = "1+2*5/3+6/4*2"
# # Output: 5


#My Approach

'''
Intuition:

    1. Process the string to make valid expression elements.
    2. Process each operator:
        - '/*-+' in that order, until there is none left.
        - Take each operator and the element to the left and to the right to compose a new element to insert it 
            where the left one where.
    3. Return the result.

'''

# def calculate(s: str) -> int:
    
#     # Handle no operators case
#     if not any(op in s for op in '/*-+'):
#         return int(s)
    

#     # Process the String to make it a valid Expression List
#     expression = []
#     num = ''

#     for char in s:

#         if char != ' ':

#             if char in '+-*/':
#                 expression.append(num)
#                 expression.append(char)
#                 num = ''
            
#             else:
#                 num += char

#     expression.append(num)  # Append the last number in the string


#     # Process the '*' and the '/' in the expression list until there are no more of those operators
#     while any(op in expression for op in '*/'):

#         for elem in expression:

#             if elem == '*':
#                 idx = expression.index('*')
#                 new_element = int(expression[idx-1]) * int(expression[idx+1])
#                 expression = expression[:idx-1] + [new_element] + expression[idx+2:]
            
#             elif elem == '/':
#                 idx = expression.index('/')
#                 new_element = int(expression[idx-1]) // int(expression[idx+1])
#                 expression = expression[:idx-1] + [new_element] + expression[idx+2:]

    
#     # Process the '+' and the '-' in the expression list until there are no more of those operators
#     while any(op in expression for op in '+-'):

#         for elem in expression:
                                        
#             if elem == '+':
#                 idx = expression.index('+')
#                 new_element = int(expression[idx-1]) + int(expression[idx+1])
#                 expression = expression[:idx-1] + [new_element] + expression[idx+2:]
            
#             elif elem == '-':
#                 idx = expression.index('-')
#                 new_element = int(expression[idx-1]) - int(expression[idx+1])
#                 expression = expression[:idx-1] + [new_element] + expression[idx+2:]


#     # Return the result
#     return expression[0]

# print(calculate(s=s))


'''
Notes: 
    This approach met 97% of the cases and it only breaks by time-limit.
'''


# Stack Approach

# import math

# def calculate(s:str) -> int:

#     num = 0
#     pre_sign = '+'
#     stack = []

#     for char in s+'+':

#         if char.isdigit():
#             num = num*10 + int(char)

#         elif char in '/*-+':

#             if pre_sign == '+':
#                 stack.append(num)
            
#             elif pre_sign == '-':
#                 stack.append(-num)
                        
#             elif pre_sign == '*':
#                 stack.append(stack.pop()*num)            
            
#             elif pre_sign == '/':
#                 stack.append(math.trunc(stack.pop()/num))
            
#             pre_sign = char
#             num = 0
    
#     return sum(stack)

# print(calculate(s=s))

'Done'




'''230. Kth Smallest Element in a BST'''

#Base

# # Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# # Case 1
# tree_layout = [3,1,4,None,2]

# one, four = TreeNode(val=1, right=TreeNode(val=2)), TreeNode(val=4)
# root = TreeNode(val=3, left=one, right=four)

# k = 1
# # Output: 1

# # Case 2
# tree_layout = [5,3,6,2,4,None,None,1]

# three, six = TreeNode(val=3, left=TreeNode(val=2, left=TreeNode(val=1)), right=TreeNode(val=4)), TreeNode(val=6)
# root = TreeNode(val=5, left=three, right=six)

# k = 3
# # Output: 3

# # Custom Case
# tree_layout = [5,3,6,2,4,None,None,1]

# three, six = TreeNode(val=3, left=TreeNode(val=2, left=TreeNode(val=1)), right=TreeNode(val=4)), TreeNode(val=6)
# root = TreeNode(val=5, left=three, right=six)

# k = 3
# # Output: 3


# My Aprroach

'''
Intuition:
    - Traverse the Tree with preorder to extract the values
    - Create a Max heap of length k and go through the rest of the elements (mantaining the heap property).
    - Return the first element of the heap.
'''

# def kth_smallest(root: TreeNode,k: int) -> int:

#     # Define Aux Inorder traversal func
#     def inorder(root: TreeNode, path:list) -> list:

#         if root:

#             node = root

#             inorder(root=node.left, path=path)
#             path.append(node.val)
#             inorder(root=node.right, path=path)

#             return path

#     tree_list = inorder(root=root, path=[])

#     tree_list.sort()

#     return tree_list[k-1]


# print(kth_smallest(root=root, k=k))

'''Notes: 
- This approach works perfectly, and it beated 37% of solutions in Runtime and 80% in space.
    
    Complexity:
    - Time complexity: O(nlogn).
    - Space Complexity: O(n).

Now, if no sorting func is required to be used, below will be that version.
'''

# # Without Sorting Approach
# import heapq

# def kth_smallest(root: TreeNode,k: int) -> int:

#     # Define Aux Inorder traversal func
#     def inorder(root: TreeNode, path:list) -> list:

#         if root:

#             node = root

#             inorder(root=node.left, path=path)
#             path.append(node.val)
#             inorder(root=node.right, path=path)

#             return path

#     # Extract the tree nodes values in a list
#     tree_list = inorder(root=root, path=[])


#     # Make a min-heap out of the tree_list up to the 'k' limit
#     heap = tree_list[:k]
#     heapq.heapify(heap)

#     # Iterate through each element in the tree_list starting from 'k' up to len(tree_list)
#     for num in tree_list[k:]:

#         if num < heap[0]:
#             heapq.heappop(heap)
#             heapq.heappush(heap, num)
    
#     return heap[-1] # The result is the last element of the min-heap, since it was length k, and the last is the kth


# print(kth_smallest(root=root, k=k))

'''Notes: 
- This approach also worked smoothly, and it consequentially reduced its performance
    beating only 6% of solutions in Runtime and it maintains the 80% in space.
    
    Complexity:
    - Time complexity: O(n+(n-k)logk).
    - Space Complexity: O(n).

Now, what if I don't traverse the elements (O(n)) and later I traverse up to k?
    Would it be possible to order the heap while traversing the tree?.
'''

# # Another enhanced solution
# import heapq

# def kth_smallest(root: TreeNode, k: int) -> int:

#     # Define the heap with 'inf' as it first element (To be pushed later on)
#     heap = [float('inf')]

#     # Define Aux Inorder traversal func
#     def inorder(root: TreeNode) -> None:

#         if root:

#             node = root

#             inorder(root=node.left)

#             if len(heap) == k:

#                 if node.val < heap[0]:
#                     heapq.heappop(heap)
#                     heapq.heappush(heap, node.val)
#                     pass
            
#             else:
#                 heap.append(node.val)


#             inorder(root=node.right)
    
#     inorder(root=root)
    
#     return heap[-1] # The result is the last element of the min-heap, since it was length k, and the last is the kth


# print(kth_smallest(root=root, k=k))

'''Notes: 
- This approach also worked smoothly, and it actually beated the first approach in performance,
    beating 57% of solutions in Runtime and it maintains the 80% in space.
    
    Complexity:
    - Time complexity: O(nlogk).
    - Space Complexity: O(n+k).

That was a great exercise, now what is the customary solution for this?.
    Quick answer: Simply inorderlt traverse the tree up to k, since is a Binary Search Tree, it was already sorted.
'''

'Done'




'''234. Palindrome Linked List'''

#Base
# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# Input
# # Case 1
# head_layout = [1,2,2,1]
# head = ListNode(val=1, next=ListNode(val=2, next=ListNode(val=2, next=ListNode(val=1))))
# # Output: True

# # Case 2
# head_layout = [1,2]
# head = ListNode(val=1, next=ListNode(val=2))
# # Output: False

# # Custom Case
# head_layout = [1,0,0]
# head = ListNode(val=1, next=ListNode(val=0, next=ListNode(val=0)))
# # Output: False


# My Approach (Brute forcing)
'''
Intuition:
    - Traverse the list collecting the values
    - Return the test that the values collected are equal to their reverse
'''

# def is_palindrome(head:ListNode) -> bool:
    
#     # Define values holder
#     visited = []    

#     # Traverse the list
#     while head:
        
#         visited.append(head.val)

#         head = head.next

#     return visited == visited[::-1]

# print(is_palindrome(head=head))

'''Note: 
    This is the most "direct" way to solve it, but there are two more way to solve this same challenge
    One involves recursion/backtracking and the other solve the problem with O(1) of space complexity, while this and
    The recursive approaches consumes O(n).'''




# Recursive Approach
'''
Intuition:
    - Make a pointer to the head of the llist (will be used later).
    - Define the Auxiliary recursive function:
        + This function will go in depth through the list and when it hits the end,
            it will start to go back in the call stack (which is virtually traversing the list backwards).
        + When the reverse traversing starts compare each node with the pointer defined at the begining and if they have equal values
            it means up to that point the palindrome property exist, otherwise, return False.
        + If the loop finishes, it means the whole list is palindromic.
    - return True.
'''
# class Solution:

#     def __init__(self) -> None:
#         pass

#     def is_palindrome(self, head:ListNode) -> bool:

#         self.front_pointer = head

#         def rec_traverse(current_node:ListNode) -> bool:

#             if current_node is not None:
                
#                 if not rec_traverse(current_node.next):
#                     return False
                
#                 if self.front_pointer.val != current_node.val:
#                     return False
            
#                 self.front_pointer = self.front_pointer.next

#             return True
        
#         return rec_traverse(head)
    
# solution = Solution()
# print(solution.is_palindrome(head=head))
'The solution as a -standalone function- is more complex than as a class method'


# Iterative Approach / Memory-efficient
'''
Intuition:
    - Use a two-pointer approach to get to the middle of the list.
    - Reverse the next half (from the 'slow' pointer) of the llist.
    - Initiate a new pointer to the actual head of the llist and in a loop (while 'the prev node')
        compare the two pointer up until they are different or the 'prev' node gets to None.
    - If the loop finishes without breaking, return 'True'.
'''

# def is_palindrome(head:ListNode) -> bool:

#     # Hanlde corner cases:
#     if not head or not head.next:
#         return True
    

#     # Traverse up to the middle of the llist
#     slow = fast = head

#     while fast and fast.next:
#         slow = slow.next
#         fast = fast.next.next

    
#     # Reverse the remaining half of the llist
#     prev = None

#     while slow:
#         next_node = slow.next
#         slow.next = prev
#         prev = slow
#         slow = next_node


#     # Compare the reversed half with the actual first half of the llist
#     left, right = head, prev

#     while right:

#         if left.val != right.val:
#             return False
        
#         left, right = left.next, right.next

    
#     # If it didn't early end then means the llist is palindromic
#     return True


# print(is_palindrome(head=head))
'Done'




'''237. Delete Node in a Linked List'''

# # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None


# # Input

# # Case 1
# llist = [4,5,1,9]
# node = ListNode(5)

# head = ListNode(4)
# head.next = node
# node.next = ListNode(1)
# node.next.next = ListNode(9)

# Output: [4,1,9]

# # Case 2
# llist = [4,5,1,9]
# node = ListNode(1)

# head = ListNode(4)
# head.next = ListNode(5)
# head.next.next = node
# node.next = ListNode(9)

# # Output: [4,5,9]


# Solution

'''
Intuition:
    - The only way to modify the list in place without accessing the head of the list is to overwrite
        the value of the given node with the next, and when reach the end, point the last node to None.
'''

# def delete_node(node:ListNode) -> None:
#     node.val = node.next.val
#     node.next = node.next.next

# #Testing
# delete_node(node=node)

# new_node = head

# while new_node:
#     print(new_node.val, end=' ')
#     new_node = new_node.next

'Done'




'''238. Product of Array Except Self'''

# Input

# # Case 1
# nums = [1,2,3,4]
# # Output: [24,12,8,6]

# # Case 2
# nums = [-1,1,0,-3,3]
# # Output: [0,0,9,0,0]


# My Approach - (Brute forcing)
# from math import prod

# def product_except_self(nums:list[int]) -> list[int]:

#     res = []

#     for i in range(len(nums)):
#         res.append(prod(nums[:i]+nums[i+1:]))

#     return res

# print(product_except_self(nums=nums))

'Note: This solution suffices 75% of test cases, but resulted inefficient with large inputs'


# # My Approach v2 - (Trying to carry the result - pseudo-prefix sum)
# from math import prod

# def product_except_self(nums:list[int]) -> list[int]:

#     res = []
#     nums_prod = prod(nums)

#     for i in range(len(nums)):

#         elem = nums_prod//nums[i] if nums[i] != 0 else prod(nums[:i]+nums[i+1:])
#         res.append(elem)


#     return res

# print(product_except_self(nums=nums))

'Note: It worked and beated 85% in time compl. and 74% in memory compl.'


# Preffix-Suffix product Approach - Customary

'''
Intuition:
    - The core idea of this approach is to build an array of the carrying product of all elements from left to right (Preffix)
        and build another more array with the sabe but from right to left (suffix).

    - After having that by combining those products but element by element, the preffix from 0 to n-1 indexed and the suffix from n-1 to 0 indexed
        and EXCLUDING the current index (i) in the final traversal, the 'self' element is explicitly excluded from the product.
'''

# from itertools import accumulate
# import operator

# def product_except_self(nums:list[int]) -> list[int]:

#     res = []

#     # Populate both preffix and suffix
#     preffix = list(accumulate(nums, operator.mul))
#     suffix = list(accumulate(reversed(nums), operator.mul))[::-1]


#     # Combine the results
#     for i in range(len(preffix)):

#         if 0 < i < len(preffix)-1:
#             res.append(preffix[i-1]*suffix[i+1])
        
#         elif i == 0:
#             res.append(suffix[i+1])

#         else:
#             res.append(preffix[i-1])
    
#     return res

# print(product_except_self(nums=nums))

'Done'




'''239. Sliding Window Maximum'''

# Input

# # Case 1
# nums = [1,3,-1,-3,5,3,6,7]
# k = 3
# # Output: [3,3,5,5,6,7]

# # Case 2
# nums = [1]
# k = 1
# # Output: [1]

# # Cusom Case
# nums = [1,3,-1,-3,5,3,6,7]
# k = 3
# # Output: [3,3,5,5,6,7]


'My approach'
# def max_sliding_window(nums:list[int], k:int) -> list[int]:

#     if len(nums) == 1:
#         return nums
    
#     if k == len(nums):
#         return [max(nums)]


#     result = []

#     for i in range(len(nums)-k+1):
#         result.append(max(nums[i:i+k]))

#     return result

# print(max_sliding_window(nums=nums, k=k))

'Note: This approach cleared 73% of test cases, but breaks with large inputs'


'Monotonically Decreacing Queue'
# def max_sliding_window(nums:list[int], k:int) -> list[int]:

#     import collections

#     output = []
#     deque = collections.deque() # nums
#     left = right = 0

#     while right < len(nums):

#         # Pop smaller values from de deque
#         while deque and nums[deque[-1]] < nums[right]:
#             deque.pop()

#         deque.append(right)

#         # remove the left val from the window
#         if left > deque[0]:
#             deque.popleft()

#         if (right+1) >= k:
#             output.append(nums[deque[0]])
#             left += 1
        
#         right += 1

#     return output

# print(max_sliding_window(nums=nums, k=k))

'done'




'''240. Search a 2D Matrix II'''

# Input

# # Case 1
# matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
# target = 5
# # Output: True

# # Case 2
# matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
# target = 20
# # Output: False


'My approach'

'''
Intuition:
    - iterativelly search in the first row of the matrix if the value is in there by a belonging test
        - If the value is in the element, break and return True / else, pop that element from the matrix
        - Transpose the matrix and start over until there's no more elements in the matrix
    - If the loop reaches the last element of the matrix, return False
'''

# def search_matrix(matrix:list[list[int]], target: int) -> bool:

#     m = len(matrix)
#     n = len(matrix[0])

#     # Handle the corner case
#     if n == m == 1:
#         return target == matrix[0][0]

#     while matrix:

#         # print(matrix)

#         element = matrix.pop(0)

#         if target in element:
#             return True
        
#         matrix = [list(x) for x in zip(*matrix)]
    
#     return False

# print(search_matrix(matrix=matrix,target=target))

'''Note: This approach doesn't worked because dinamically changing the data structure mess in how python checks membership'''


'''Binary search approach'''

# def search_matrix(matrix:list[list[int]], target: int) -> bool:

    # m = len(matrix)
    # n = len(matrix[0])

    # # Handle the corner case
    # if n == m == 1:
    #     return target == matrix[0][0]

    # row, col = m-1, 0   # Start the search from the bottom left corner

    # while row >= 0 and col < n:

    #     element = matrix[row][col]

    #     if element == target:
    #         return True
        
    #     elif element > target:
    #         row -= 1
                   
    #     else:
    #         col += 1
    
    # return False

# print(search_matrix(matrix=matrix,target=target))

'done'




'''279. Perfect Squares'''

# Input

# # Case 1
# n = 12
# # Output: 3 (4+4+4)

# # Case 2
# n = 13
# # Output: 2 (4+9)

# # Custom case
# n = 43
# # Output: 3 

# # Custom case
# n = 67
# # Output: 3 

'My approach'

'''
Intuition:
    - Build the possible addends (Each number that its 2nd power is less than n).
    - Reverse the addends (To have them from bigger to smaller).
    - Iteratively check from bigger to smaller how many addends can be grouped to sum up using modulo and division.
        + If a division if the group is still short reach the n, go to the next addend to fill up.
    - When the group is completed, start the process over but starting from the next addend.
        The last group will always be the largest, since it consists of a groups of 1's.
    - Return the count of the shortest group.
'''

# def num_squares(n:int) -> int:

#     # Define the holder of the groups
#     result = []

#     # Define the holder and the indext to populate the addends
#     addends = []
#     i = 1

#     # Populate the addends / ex: [1, 4, 9]
#     while i*i <= n:
#         addends.append(i*i)
#         i += 1

#     # Reverse the addends
#     addends = addends[::-1]

#     # Form the groups
#     for i in range(len(addends)):

#         group = []

#         for j in range(i, len(addends)):

#             if sum(group) == n:
#                 break
        
#             if (n-sum(group))/addends[j] >= 1:                
#                 group += ([addends[j]] * ((n-sum(group))//addends[j]))
        
#         result.append(group) if len(group) != n else None

#     # Sort the groups from the shortest to the largest
#     result.sort(key=len)

#     #return the shortest
#     return len(result[0])

# print(num_squares(n=n))

'This solution cleared 96% of the test cases, the actual DP solution didnt made sense to me'




'''283. Move Zeroes'''

# Input

# # Case 1
# nums = [0,1,0,3,12]
# # Output: [1,3,12,0,0]

# # Case 2
# nums = [0]
# # Output: [0]

# # Custom Case
# nums = [2,3,4,0,5,6,8,0,1,0,0,0,9]
# # Output: [0]


'My approach'

'''
Intuition:
    - Create a new list as a buffer to hold every item in the initial order
    - Separate the buffer into non-zeroes and zeroes different list and joint them together.
    - Replace each value of the original list with the order or the buffer list.

This solution is more memory expensive than one with a Two-pointer approach, but let's try it
'''

# def move_zeroes(nums:list[int]) -> None:

#     # Handle corner case
#     if len(nums) == 1:
#         return nums
 
#     # Create the buffers to separate the non-zeroes to the zeroes
#     non_zeroes, zeroes = [x for x in nums if x != 0],[x for x in nums if x == 0]

#     # Join the buffers into one single list
#     buffer = non_zeroes + zeroes

#     # Modify the original input with the buffer's order
#     for i in range(len(nums)):
#         nums[i] = buffer[i]
    
# move_zeroes(nums=nums)

# print(nums)

'Note: This solution was accepted and beated submissions by 37% in runtime and 87% in memory'


'Two-pointers Approach'
# def move_zeroes(nums:list[int]) -> None:

#     # Initialize the left pointer
#     l = 0

#     # Iterate with the right pointer through the elements of nums
#     for r in range(len(nums)):

#         if nums[r] != 0:

#             nums[r], nums[l] = nums[l], nums[r]

#             l += 1

# move_zeroes(nums=nums)

# print(nums)

'Done'




'''287. Find the Duplicate Number'''

# Input

# # Case 1
# nums = [1,3,4,2,2]
# # Output: 2

# # Case 2
# nums = [3,1,3,4,2]
# # Output: 3

# # Custom Case
# nums = [3,3,3,3,3]
# # Output: 3

'My approach'

# def find_duplicate(nums:list[int]) -> int:

#     for num in nums:

#         if nums.count(num) != 1:
#             return num

# print(find_duplicate(nums=nums))

'Note: This approach cleared 92% of cases but breaks with larger inputs'


'Hare & Tortoise Approach'

# def find_duplicate(nums:list[int]) -> int:

#     # Initialize two pointers directing to the first element in the list
#     slow = fast = nums[0]

#     # Iterate until they coincide (They' found each other in the cycle)
#     while True:
#         slow = nums[slow]
#         fast = nums[nums[fast]]
        
#         if slow == fast:
#             break
    
#     # Reset the slow to the begining of the list, so they an meet at the repeating number
#     slow = nums[0]

#     # Iterate again but at same pace, they will eventually meet at the repeated number
#     while slow != fast:
#         slow = nums[slow]
#         fast = nums[fast]

#     return fast

# print(find_duplicate(nums=nums))

'Done'




'''xxx'''























