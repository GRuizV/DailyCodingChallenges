'''
CHALLENGES INDEX

297. Serialize and Deserialize Binary Tree (BFS)
300. Longest Increasing Subsequence (DP)
315. Count of Smaller Numbers After Self - Partially solved
322. Coin Change (DP)
326. Power of Three (RC)
328. Odd Even Linked List
329. Longest Increasing Path in a Matrix (Matrix) (DFS) (MEM)
334. Increasing Triplet Subsequence (GRE)
344. Reverse String (TP)
350. Intersection of Two Arrays II (TP)
341. Flatten Nested List Iterator (DFS)
347. Top K Frequent Elements (Heaps) (Sorting)
378. Kth Smallest Element in a Sorted Matrix (Matrix) (Heaps)
380. Insert Delete GetRandom O(1)
384. Shuffle an Array
395. Longest Substring with At Least K Repeating Characters (SW) (RC) (DQ)








*DP: Dynamic Programming
*RC: Recursion
*TP: Two-pointers
*FCD: Floyd's cycle detection (Hare & Tortoise approach)
*PS: Preffix-sum
*SW: Sliding-Window
*MEM: Memoization
*GRE: Greedy
*DQ: Divide and Conquer



(16)
'''




'''297. Serialize and Deserialize Binary Tree'''

# # Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


# Input

# # Case 1
# root_map = [1,2,3,None,None,4,5]

# root = TreeNode(1)
# two, three = TreeNode(2), TreeNode(3)
# four, five = TreeNode(4), TreeNode(5)

# root.left, root.right = two, three
# three.left, three.right = four, five

# Custom Case
# root_map = [4,-7,-3,None,None,-9,-3,9,-7,-4,None,6,None,-6,-6,None,None,0,6,5,None,9,None,None,-1,-4,None,None,None,-2]

# root = TreeNode(4)
# two, three = TreeNode(-7), TreeNode(-3)
# root.left, root.right = two, three

# four, five = TreeNode(-9), TreeNode(-3)
# three.left, three.right = four, five

# six, seven, eight = TreeNode(9), TreeNode(-7), TreeNode(-4)
# four.left, four.right = six, seven
# five.left = eight

# nine, ten = TreeNode(6), TreeNode(-6)
# seven.left = nine
# eight.right = ten

# eleven, twelve, thirteen = TreeNode(-6), TreeNode(0), TreeNode(6)
# nine.left, nine.right = eleven, twelve
# ten.left = thirteen

# fourteen, fifteen = TreeNode(5), TreeNode(-2)
# thirteen.left, thirteen.right = fourteen, fifteen

# sixteen, seventeen = TreeNode(9), TreeNode(-1)
# fourteen.left, fourteen.right = sixteen, seventeen

# eighteen = TreeNode(-4)
# seventeen.left = eighteen

# Output: [4,-7,-3,null,null,-9,-3,9,-7,-4,null,6,null,-6,-6,null,null,0,6,5,null,9,null,null,-1,-4,null,null,null,-2]




'Solution'

# class Codec:

#     def serialize(self, root):
#         """
#         Encodes a tree to a single string.
        
#         :type root: TreeNode
#         :rtype: str
#         """

#         # Handle corner case
#         if not root:
#             return ''

#         queue = [root]
#         visited = []

#         while queue:

#             node = queue.pop(0)

#             if node:
#                 visited.append(str(node.val))
#                 queue.extend([node.left, node.right])

#             else:
#                 visited.append('None')        

#         return ','.join(visited)
        

#     def deserialize(self, data):
#         """
#         Decodes your encoded data to tree.
        
#         :type data: str
#         :rtype: TreeNode
#         """

#         # Handle corner case
#         if not data:
#             return
        
#         # Transform data into a valid input for the tree
#         data = [int(x) if x != 'None' else None for x in data.split(',')]

#         # Initilize the root
#         root = TreeNode(data[0])

#         # Populate the tree
#         index = 1
#         queue = [root]

#         while index < len(data) and queue:

#             node = queue.pop(0)

#             if data[index]:

#                 node.left = TreeNode(data[index])
#                 queue.append(node.left)
            
#             index += 1
            
#             if data[index]:

#                 node.right = TreeNode(data[index])
#                 queue.append(node.right)
            
#             index += 1

#         return root


# # Testing

# ser = Codec()
# deser = Codec()
# ans = deser.serialize(root=root)

# # print(ans)


# # Your Codec object will be instantiated and called as such:
# ser = Codec()
# deser = Codec()
# ans = deser.deserialize(ser.serialize(root))

# # print(ans)


# # Auxiliary pretty print function
# def pretty_print_bst(node, prefix="", is_left=True):

#     if not node:
#         node = root
    
#     if not node:
#         print('Empty Tree')
#         return


#     if node.right is not None:
#         pretty_print_bst(node.right, prefix + ("│   " if is_left else "    "), False)

#     print(prefix + ("└── " if is_left else "┌── ") + str(node.val))

#     if node.left is not None:
#         pretty_print_bst(node.left, prefix + ("    " if is_left else "│   "), True)


# pretty_print_bst(node=root)
# print('\n\n\n\n')
# pretty_print_bst(node=ans)

'Done'




'''300. Longest Increasing Subsequence'''

# # Input

# # Case 1
# nums = [10,9,2,5,3,7,101,18]
# # Output: 4

# # Case 2
# nums = [0,1,0,3,2,3]
# # Output: 4

# # Case 3
# nums = nums = [7,7,7,7,7,7,7]
# # Output: 1


'DP Solution'

# def lengthOfLIS(nums: list[int]) -> int:    
    
#     # Handle corner case
#     if not nums:
#         return 0
    

#     # Initialize the dp array
#     dp = [1] * len(nums)

#     # Iterate through the elements of the list, starting from the second
#     for i in range(1, len(nums)):

#         for j in range(i):

#             if nums[i] > nums[j]:
#                 dp[i] = max(dp[i], dp[j]+1)


#     return max(dp)

'Done'




'''315. Count of Smaller Numbers After Self'''

# Input

# # Case 1
# nums = [5,2,6,1]
# # Output: [2,1,1,0]

# # Case 1
# nums = [-1,-1]
# # Output: [0,0]


'My Approach (Brute forcing)'

# def count_smaller(nums: list[int]) -> list[int]:

#     # Handle corner case
#     if len(nums) == 1:
#         return [0]
    

#     # Set the min value of the group
#     min_num = min(nums)
    

#     # Initialize the result holder
#     result = []

#     for x,num in enumerate(nums):

#         # corner case: if the number is the smallest of the group or the right most one, no smaller numbers after it
#         if num == min_num or num == nums[-1]:
#             result.append(0)
        
#         else:

#             # Define a sublist with all elements to the right of the current one
#             sublist = nums[x+1:]

#             # Count how many of those are smaller than the current one
#             count = len([x for x in sublist if x<num])

#             # Add that result to the holder
#             result.append(count)
            
#     return result
     

# print(count_smaller(nums=nums))

'Note: This approach met up to 79% o the cases'





'''322. Coin Change'''

# Input

# # Case 1
# coins = [1,2,5]
# amount = 11
# # Output: 3

# # Case 2
# coins = [2]
# amount = 3
# # Output: -1

# # Custome Case
# coins = [186,419,83,408]
# amount = 6249
# # Output: 20


'My Approach (Greedy approach)'

# def coin_change(coins:list[int], amount: int) -> int:

#     # Handle Corner Case
#     if not amount:
#         return 0
    
#     # Sort the coins decreasingly
#     coins = sorted(coins, reverse=True)

#     # Initialize the coins counter
#     result = 0

#     # Iterate through
#     for coin in coins:

#         if coin <= amount:

#             result += amount // coin
#             amount %= coin
        
#         if not amount:
#             return result
    
#     # If the execution get to this point, it means it was not an exact number of coins for the total of the amount
#     return -1

# print(coin_change(coins=coins, amount=amount))

'Note: This is a Greedy approach and only met up 27% of test cases'


'DP Approach'

# def coin_change(coins:list[int], amount: int) -> int:

#     # DP INITIALIZATION
#     # Initialize the dp array
#     dp = [float('inf')] * (amount+1)

#     # Initialize the base case: 0 coins for amount 0
#     dp[0] = 0

#     # DP TRANSITION
#     for coin in coins:

#         for x in range(coin, amount+1):
#             dp[x] = min(dp[x], dp[x-coin] + 1)

#     # Return result
#     return dp[amount] if dp[amount] != float('inf') else -1


# print(coin_change(coins=coins, amount=amount))

'Done'





'''326. Power of Three'''

# Input

# # Case 1
# n = 45
# # Output: True

# # Custom Case
# n = -1
# # Output: True


'Iterative approach'

# def is_power_of_three(n:int) -> bool:

#     powers = [3**x for x in range(21)]

#     return n in powers



'Recursive apporach'

# def is_power_of_three(n:int) -> bool:

#     # Base case: if n is 1, it's a power of three
#     if n == 1:
#         return True

#     # If n is less than 1, it can't be a power of three
#     if n < 1:
#         return False

#     # Recursive case: check if n is divisible by 3 and then recurse with n divided by 3
#     if n % 3 == 0:
#         return is_power_of_three(n // 3)

#     # If n is not divisible by 3, it's not a power of three
#     return False

'Done'





'''328. Odd Even Linked List'''

# from typing import Optional

# Base

# # Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


# Input

# # Case 1
# list_map = [1,2,3,4,5]
# head = ListNode(1, ListNode(2, ListNode(3, ListNode(4, ListNode(5)))))
# # Output: [1,3,5,2,4]

# # Case 2
# list_map = [2,1,3,5,6,4]
# head = ListNode(2, ListNode(1, ListNode(3, ListNode(5, ListNode(6, ListNode(4))))))
# # Output: [2,3,6,1,5,4]


'My Approach'

'''
Intuition:
    - Create a mock node (even) to hold its respective members.
    - Traverse the list and each even node conect it to the mock node and the odd node conected to the next of its consequent even.
    - Conect the tail of the original node (now the odd nodes) to the mocking node 'even'.
'''

# def oddEvenList(head: Optional[ListNode]) -> Optional[ListNode]:

#     # Handle corner case: There has to be a node to do something.
#     if head:        

#         # Initialize the 'even' nodes holder
#         even = ListNode()

#         # Traverse the LList & separete odds from even
#         curr = head
#         curr_even = even
        
#         while curr:

#             if curr.next:
                
#                 # Assign the connection
#                 curr_even.next = curr.next
#                 curr.next = curr.next.next

#                 # Continue traversing
#                 curr = curr.next
#                 curr_even = curr_even.next
            
#             else:
#                 break


#         # if the lenght of the LList is odd:
#         #   The tail of odds is a node and must be connect to the even head, and the tail of even must point to None.
#         # if is even:
#         #   The tail of odd is None and the list mus be traversed again to connect its tail to even's head and the tail of even is already pointing to None.
        
#         if curr:
#             curr_even.next = None
#             curr.next = even.next

#         else:
#             curr = head

#             while curr.next:
#                 curr = curr.next
            
#             curr.next = even.next
        
#         # As the modification was in place, there is no return statement



# oddEvenList(head=head)

# # Testing
# curr = head

# while curr:
#     print(curr.val, end=' ')
#     curr = curr.next


'Note: This approached worked, it beated 71% of submissions in Runtime and 38% in Memory'



'A cleaner approach of the same'

# def oddEvenList(head):
#     if not head or not head.next:
#         return head

#     odd = head
#     even = head.next
#     even_head = even

#     while even and even.next:
#         odd.next = even.next
#         odd = odd.next
#         even.next = odd.next
#         even = even.next

#     odd.next = even_head

#     return head

'Done'





'''329. Longest Increasing Path in a Matrix'''

# Input

# # Case 1
# matrix = [[9,9,4],[6,6,8],[2,1,1]]
# # Output: 4 // Longest path [1, 2, 6, 9]

# # Case 2
# matrix = [[3,4,5],[3,2,6],[2,2,1]]
# # Output: 4 // Longest path [3, 4, 5, 6]


'My Approach (DP)'

'''
Intuition:

    Thinking in the matrix as a graph my intuition is to check each node
    following DFS for its vecinity only if the neighbor is higher than the curr node value,
    and store the possible path length from each node in a DP matrix. after traversing the graph
    the max value in the DP matrix will be the answer
'''

# def longestIncreasingPath(matrix: list[list[int]]) -> int:

#     # Handle corner case: no matrix
#     if not matrix or not matrix[0]:
#         return 0

#     # Capturing the matrix dimentions
#     m,n = len(matrix), len(matrix[0])

#     # Defining the DP matrix
#     dp = [[1]*n for _ in range(m)]

#     # Define the directions for later adding the neighbors
#     directions = [(1,0),(-1,0),(0,1),(0,-1)]
    
#     # Traverse the matrix
#     for i in range(m):

#         for j in range(n):

#             # Define its max: its current max path in the dp matrix
#             elem_max = dp[i][j]

#             # Define the actual neighbors: The element within the matrix boundaries and higher and itself
#             neighbors = [(i+dx, j+dy) for dx,dy in directions if 0<=i+dx<m and 0<= j+dy<n and matrix[i+dx][j+dy] > matrix[i][j]]

#             # Check for each neighbor's max path while redefine its own max path
#             for neighbor in neighbors:
#                 curr = dp[i][j]
#                 next_max = max(curr, curr + dp[neighbor[0]][neighbor[1]])
#                 elem_max = max(elem_max, next_max)
            
#             # Update it in the dp matrix
#             dp[i][j] = elem_max    

#     # get dp's max
#     result = max(max(x) for x in dp)
    
#     # Return its value
#     return result

# print(longestIncreasingPath(matrix=matrix))

'Note: This approach only works if it starts from the node with the largest value'


'DFS with Memoization Approach'

# def longestIncreasingPath(matrix: list[list[int]]) -> int:

#     # Handle Corner Case
#     if not matrix or not matrix[0]:
#         return 0

#     # Capture matrix's dimentions
#     m, n = len(matrix), len(matrix[0])

#     # Define the memoization table
#     dp = [[-1] * n for _ in range(m)]

#     # Define the directions
#     directions = [(1,0),(-1,0),(0,1),(0,-1)]
    
#     # Define the DFS helper function
#     def dfs(x, y):

#         # Handle corner case: the cell was already visited
#         if dp[x][y] != -1:
#             return dp[x][y]
        
#         # Define the max starting path, which is 1 for any cell
#         max_path = 1

#         # Define the directions to go
#         for dx, dy in directions:

#             nx, ny = x + dx, y + dy

#             # If it's a valid neighbor, recalculate the path
#             if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] > matrix[x][y]:
                
#                 # The new path will be the max between the existing max path and any other valid path from the neighbor
#                 max_path = max(max_path, 1 + dfs(nx, ny))
        
#         # Update the Memoization table
#         dp[x][y] = max_path
        
#         # Return the value
#         return dp[x][y]
    

#     # Define the initial max lenght
#     max_len = 0

#     # Run the main loop for each cell
#     for i in range(m):
#         for j in range(n):
#             max_len = max(max_len, dfs(i, j))
    
#     # Return the max length
#     return max_len

# print(longestIncreasingPath(matrix=matrix))

'Done'





'''334. Increasing Triplet Subsequence'''

# Input

# # Case 1
# nums = [1,2,3,4,5]
# # Output: True / Any triplet where i < j < k is valid.

# # Case 2
# nums = [5,4,3,2,1]
# # Output: False / Any triplet where i < j < k is valid.

# # Case 3
# nums = [2,1,5,0,4,6]
# # Output: True / The triplet (3, 4, 5) where [0,4,6] is valid.

# # Custom Case
# nums = [1,2,2147483647]
# # Output: False.



'My approach (Brute forcing) - Iterative looping'

'''
Intuition:
    - Handle corner cases: 
        + If no input; 
        + if input length < 3; 
        + If input length = 3 != to sorted(input, reverse = False)
        + If input == sorted(input, reverse = True)

    - In a while loop check one by one, starting from the first index, if next to it is any other element greater than it.
        from that element start the search for a greater element than the first greater and 
        
        + if found, return True;
        + else, move the initial index to the next and start over
        + if the initial index gets to the second last element and no triplet has been found, return False.
'''

# def increasingTriplet(nums: list[int]) -> bool:

#     # Handle corner cases
#     if not nums or len(nums) < 3 or (len(nums) == 3 and nums != sorted(nums, reverse=True)) or nums == sorted(nums, reverse=True):
#         return False

#     # Initialize the triplet initial index
#     i = 0

#     # Iterate through the input elements
#     while i < len(nums)-2:

#         for j in range(i+1, len(nums)):

#             if nums[j] > nums[i]:

#                 for k in range(j+1, len(nums)):

#                     if nums[k] > nums[j]:

#                         return True
                    
#         i += 1
    
#     return False

# print(increasingTriplet(nums=nums))

'''
Note: This approach met 90% of test cases, but failed with larger inputs.
    
    Time complexity O(n^3)
'''



'My approach - Iterative selection'

'''
Intuition:
    - Starting from the first index, check with listcomp if there is a larger element present.
        + if it does, get its index and do the same but for this second element.
            * if there are a larger element present return True,
            * else, move the initial input to the next and start over.

    - Like the prior approach if it reaches the second last element in the input, end the loop and return False
'''

# def increasingTriplet(nums: list[int]) -> bool:

#     # Handle corner cases
#     # if not nums or len(nums) < 3 or (len(nums) == 3 and nums != sorted(nums, reverse=True)) or nums == sorted(nums, reverse=True):
#     #     return False

#     # Initialize the triplet initial index
#     i = 0

#     # Iterate through the input elements
#     while i < len(nums)-2:

#         # Get the next greater element of nums[i]
#         sec_greater = list(filter(lambda x: x>nums[i], nums[i+1:-1]))

#         # if such element exist
#         if sec_greater:    
            
#             # Iterate again for the rest of the greater elements
#             for elem in sec_greater:

#                 # Get the idx of the first greater element than nums[i]
#                 j = nums.index(elem, i+1)            

#                 # Find a element greater than nums[j]
#                 third_greater = list(filter(lambda x: x>nums[j], nums[j+1:]))

#                 # if there are greater element than nums[j], return True
#                 if third_greater:
#                     return True       
                        
#         i += 1
    
#     return False


# print(increasingTriplet(nums=nums))


'''
Note: This approach met 90% of test cases, but failed with larger inputs.
    
    Time complexity O(n^2*logn)
'''



'Optimized solution O(n)'

# def increasingTriplet(nums: list[int]) -> bool:

#     first = float('inf')
#     second = float('inf')
    
#     for num in nums:

#         if num <= first:
#             first = num

#         elif num <= second:
#             second = num

#         else:
#             return True
    
#     return False

# print(increasingTriplet(nums=nums))

'Done'





'''344. Reverse String'''

'Note: The problem ask for modify in place an iterable'

'Two pointers approach'

# def reverse_string(s: list[str]) -> None:

#     left, right = 0, len(s)

#     while left < right:

#         s[left], s[right] = s[right], s[left]

#         left += 1
#         right -= 1

'Done'





'''350. Intersection of Two Arrays II'''

# Input

# # Case 1
# nums1, nums2 = [1,2,2,1], [2,2]
# # Output: [2,2]

# # Case 2
# nums1, nums2 = [4,9,5], [9,4,9,8,4]
# # Output: [4,9]


'My approach'

'''
Intuition:
    - Handle a corner case.
    - Make a list holder for the result.
    - Get the largest list.
    - Collect the common elements and populate the result holder with the lower count from both inputs.
'''

# def intersect(nums1: list[int], nums2: list[int]) -> list[int]:

#     # Handle corner case
#     if not nums1 or not nums2:
#         return []

#     # Create a list holder for the common elements
#     commons = []

#     # Create an iterator with the longest list
#     longest = nums1 if len(nums1)>len(nums2) else nums2
    
#     # Collect the common elements
#     for elem in longest:

#         if elem in nums1 and elem in nums2 and elem not in commons:

#            count = nums1.count(elem) if nums1.count(elem) < nums2.count(elem) else nums2.count(elem)

#            commons.extend([elem]*count)
    
#     return commons

# print(intersect(nums1=nums1, nums2=nums2))


'Note: This approach worked and beated only 5% in runtine and 93% in memory'


'Two pointer approach'

# def intersect(nums1: list[int], nums2: list[int]) -> list[int]:

#     # Sort both arrays
#     nums1.sort()
#     nums2.sort()
    
#     # Initialize pointers and the result list
#     i, j = 0, 0
#     result = []
    
#     # Traverse both arrays
#     while i < len(nums1) and j < len(nums2):

#         if nums1[i] < nums2[j]:
#             i += 1

#         elif nums1[i] > nums2[j]:
#             j += 1

#         else:
#             result.append(nums1[i])
#             i += 1
#             j += 1
    
#     return result

# print(intersect(nums1=nums1, nums2=nums2))

'Done'





'''341. Flatten Nested List Iterator'''

# Base

"""
This is the interface that allows for creating nested lists.
You should not implement it, or speculate about its implementation
"""
# class NestedInteger:
#    def isInteger(self) -> bool:
#        """
#        @return True if this NestedInteger holds a single integer, rather than a nested list.
#        """

#    def getInteger(self) -> int:
#        """
#        @return the single integer that this NestedInteger holds, if it holds a single integer
#        Return None if this NestedInteger holds a nested list
#        """

#    def getList(self) -> None: #[NestedInteger] is the actual expected return
#        """
#        @return the nested list that this NestedInteger holds, if it holds a nested list
#        Return None if this NestedInteger holds a single integer
#        """


# Input

# # Case 1
# nested_list = [[1,0],2,[1,1]]
# # Output: [1,1,2,1,1]

# # Case 2
# nested_list = [1,[4,[6]]]
# # Output: [1,4,6]


'The Solution'

# class NestedIterator:

#     def __init__(self, nestedList: list[NestedInteger]):
    
#         # Initialize the stack with the reversed nested list
#         self.stack = nestedList[::-1]
    
#     def next(self) -> int:

#         # The next element must be an integer, just pop and return it
#         return self.stack.pop().getInteger()
    
#     def hasNext(self) -> bool:

#         # While there are elements in the stack and the top element is an Integer to be returned
#         while self.stack:
            
#             # Peek at the top element
#             top = self.stack[-1]
            
#             # If it's an integer, we're done
#             if top.isInteger():
#                 return True
            
#             # Otherwise, it's a list, pop it and push its contents onto the stack
#             self.stack.pop()
#             self.stack.extend(top.getList()[::-1])
        
#         # If the stack is empty, return False
#         return False

'Done'





'''347. Top K Frequent Elements'''

# Input

# # Case 1
# nums = [1,2,2,1]
# k = 2
# # Output: [1,2]

# # Case 2
# nums = [1]
# k = 1
# # Output: [1]


'My approach'

'''
Intuition:
    
Ideas' pool:
    + A Counter function approach:
        - Call a Counter on the input and sort by freq, return in order.

    + A Heap approach:
        - ...    

'''

# def topKFrequent(nums: list[int], k: int) -> list[int]:

#     # Create the result list holder
#     result = []

#     # Import Counter
#     from collections import Counter

#     #  Transform the input into a list sorted by freq
#     nums = sorted(Counter(nums).items(), key=lambda x: x[1], reverse=True)

#     # Populate the result accordingly
#     for i in range(k):
#         result.append(nums[i][0])

#     # Return the result
#     return result

# # Testing
# print(topKFrequent(nums=nums, k=k))

'Note: This approach worked beating submissions only 20% in Runtime and 61% in Memory'

'Done'





'''378. Kth Smallest Element in a Sorted Matrix'''

# Input

# # Case 1
# matrix = [[1,5,9],[10,11,13],[12,13,15]]
# k = 8
# # Output: 13 / The elements in the matrix are [1,5,9,10,11,12,13,13,15], and the 8th smallest number is 13

# # Case 2
# matrix = matrix = [[-5]]
# k = 1
# # Output: -5

'My approach'

'''
Intuition:

    Ideas' pool:

        + Brute forcing: flatten the input, sort and return
        + Heap: Hold a min heap of size k, traverse all items in the matrix and return the last of the heap

'''


'Brute force'

# def kthSmallest(matrix: list[list[int]], k: int) -> int:

#     # Flatten the input
#     matrix = [x for elem in matrix for x in elem]

#     # Sort the resulting matrix
#     matrix.sort()

#     # x=0 

#     # Return the kth element
#     return matrix[k-1]

# print(kthSmallest(matrix=matrix, k=k))

'Note: This approach works, it has O(nlongn) time complexity and beated other submissions by 89% in Runtine and 22% in Memory'


'Min-heap approach'

# def kthSmallest(matrix: list[list[int]], k: int) -> int:

#     # Capture the matrix dimentions
#     n = len(matrix)

#     # Import the heapq module
#     import heapq
    
#     # Create a min-heap with the first element of each row
#     min_heap = [(matrix[i][0], i, 0) for i in range(n)]
#     heapq.heapify(min_heap)
    
#     # Extract min k-1 times to get to the kth smallest element
#     for _ in range(k - 1):
#         value, row, col = heapq.heappop(min_heap)
#         if col + 1 < n:
#             heapq.heappush(min_heap, (matrix[row][col + 1], row, col + 1))
    
#     # The root of the heap is the kth smallest element
#     return heapq.heappop(min_heap)[0]

# print(kthSmallest(matrix=matrix, k=k))

'Note: This solution worked, it has a time complexity of O(klogn) and beated submissions by 50% in Runtime and 34% in Memory.'

'Done'





'''380. Insert Delete GetRandom O(1)'''

# Input

# # Case 1
# operations = ["RandomizedSet", "insert", "remove", "insert", "getRandom", "remove", "insert", "getRandom"]
# inputs = [[], [1], [2], [2], [], [1], [2], []]
# # Output: [None, true, false, true, 2, true, false, 2]

'My approach'

# import random

# class RandomizedSet:

#     def __init__(self):
#         self.set: set = set()
#         print('Set created!')
        

#     def insert(self, val: int) -> bool:

#         if val not in self.set:
#             self.set.add(val)
#             return True

#         else:
#             return False
        

#     def remove(self, val: int) -> bool:

#         if val in self.set:
#             self.set.remove(val)
#             return True
        
#         else:
#             return False
        

#     def getRandom(self) -> int:

#         return random.choice(list(self.set))

'Note: While this approach works, it has O(1) time complexity for all the functions, the list casting in the getRandom() function make it go up to O(n) breaking the challenge requirement'


'An optimal solution'

# import random

# class RandomizedSet:

#     def __init__(self):
#         self.list = []
#         self.dict = {}
                
#     def insert(self, val: int) -> bool:

#         if val in self.dict:
#             return False
        
#         self.dict[val] = len(self.list)
#         self.list.append(val)

#         return True

#     def remove(self, val: int) -> bool:

#         if val not in self.dict:
#             return False
        
#         last_value, idx = self.list[-1], self.dict[val]

#         # Rewrite the list and the dict
#         self.list[idx], self.dict[last_value] = last_value, idx

#         # Update the list to remove the duplicate
#         self.list.pop()

#         # Remove the value entry in the dict
#         del self.dict[val]

#         return True
        
#     def getRandom(self) -> int:

#         return random.choice(self.list)

# # Testing
# for i, op in enumerate(operations):

#     if op == 'RandomizedSet':
#         obj = RandomizedSet()
        
#     elif op == 'insert':
#         print(obj.insert(inputs[i][0]))
    
#     elif op == 'remove':
#         print(obj.remove(inputs[i][0]))
    
#     else:
#         print(obj.getRandom())

'Done'





'''384. Shuffle an Array'''

# Input

# # Case 1
# operations = ["Solution", "shuffle", "reset", "shuffle"]
# inputs = [[[1, 2, 3]], [], [], []]
# # Output: [None, [3, 1, 2], [1, 2, 3], [1, 3, 2]]

# # Case 2
# operations = ["Solution","reset","shuffle","reset","shuffle","reset","shuffle","reset","shuffle"]
# inputs = [[[-6,10,184]],[],[],[],[],[],[],[],[]]
# # Output: [[null,[-6,10,184],[-6,10,184],[-6,10,184],[-6,184,10],[-6,10,184],[10,-6,184],[-6,10,184],[10,-6,184]]


'My approach'

# import random

# class Solution:

#     def __init__(self, nums: list[int]):
#         self.base = nums
              
#     def reset(self) -> list[int]:       
#         return self.base
        
#     def shuffle(self) -> list[int]:
#         nums = self.base[:] # Deepcopy of the list
#         random.shuffle(nums)        
#         return nums
    

# # Testing
# for i, op in enumerate(operations):

#     if op == 'Solution':
#         obj = Solution(inputs[i][0])
#         print('Object created')

#     elif op == 'shuffle':
#         print(obj.shuffle())

#     else:
#         print(obj.reset())

'Notes: This approached worked beating 88% of submissions in runtime and 25% in memory'

'Done'





'''395. Longest Substring with At Least K Repeating Characters'''

# Input

# # Case 1
# s, k = "aaabb", 3
# # Output: 3 / The longest substring is "aaa", as 'a' is repeated 3 times.

# # Case 2
# s, k = "ababbc", 2
# # Output: 5 / The longest substring is "aaa", as 'a' is repeated 3 times.


'My approach'

'''
Intuition:
    
    Brute forcing:

        - Import the Counter class from collections.
        - Initialize a max_len counter in 0 to hold the max len of a valid substring according to the requirements of k.
        - Starting from the len(s) down to k, check in a range, all the substrings of all those different sizes and
            with Counter's help check is the minimum freq is at least k,
                if it does: Refresh the max_len counter.
                if it doesn't: check the rests of the substrings

'''

# from collections import Counter

# def longestSubstring(s: str, k: int) -> int:

#     # Initialize the max counter
#     max_len = 0

#     # Capture the len of s
#     l = len(s)

#     # Handle the corner case: len(s) < k
#     if l < k:
#         return max_len

#     # Check all possibles valid substrings
#     for i in range(k-1, l):

#         for j in range(l-i):

#             # Create the possible valid substring
#             substring = s[j:j+i+1]

#             # Create a counter from the substring
#             subs_counter = Counter(substring)

#             # Capture the minimum freq of the caracters present
#             subs_min_freq = min(subs_counter.values())

#             # Update the counter only if the minimum is at least k in size
#             max_len = len(substring) if subs_min_freq >= k else max_len


#     # Return what's un the max counter
#     return max_len

# Testing
# print(longestSubstring(s=s, k=k))

'Note: This approach met the 87% of cases but with large input breaks. I will rethink the loop to make it go from the largest to the lowest limit, that should save some runtime.'


'My 2nd approach'

# from collections import Counter

# def longestSubstring(s: str, k: int) -> int:

#     # Capture the len of s
#     l = len(s)

#     # Handle the corner case: len(s) < k
#     if l < k:
#         return 0

#     # Check all possibles valid substrings
#     for i in range(l-1, k-2, -1):

#         if i != -1:

#             for j in range(l-i):
                        
#                 # Create the possible valid substring
#                 substring = s[j:j+i+1]

#                 # Create a counter from the substring
#                 subs_counter = Counter(substring)

#                 # Capture the minimum freq of the caracters present
#                 subs_min_freq = min(subs_counter.values())

#                 # If the min freq found is at least k, that's the longest valid substring possible
#                 if subs_min_freq >= k:
#                     return len(substring)

#     # Return 0
#     return 0

# # Testing
# print(longestSubstring(s=s, k=k))

'Note: Unfortunately my second approach had the same performance.'


'Divide and Conquer approach'

from collections import Counter

def longestSubstring(s: str, k: int) -> int:

    # Base case
    if len(s) == 0 or len(s) < k:
        return 0

    # Count the frequency of eachcharacter in the string
    counter = Counter(s)

    # Iterate through the string and split at a character that doesn't meet the frequency requirement
    for i, char in enumerate(s):

        if counter[char] < k:

            # Split and recursively process the left and right substrings
            left_part = longestSubstring(s[:i], k)
            right_part = longestSubstring(s[i+1:], k)

            return max(left_part, right_part)

    # If there's no splits, means that the entire substring is valid
    return len(s)

'Done'





'''xxx'''














