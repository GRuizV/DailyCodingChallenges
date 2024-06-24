'''
CHALLENGES INDEX

179. Largest Number
189. Rotate Array  (TP)
198. House Robber (DS)
200. Number of Islands  (Matrix) (BFS) (DFS)
202. Happy Number (FCD) (TP)
204. Count Primes
206. Reverse Linked List
207. Course Schedule (DFS)





*DS: Dynamic Programming
*RC: Recursion
*TP: Two-pointers
*FCD: Floyd's cycle detection


(8)
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


# # My approach

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




'''xxx'''

























