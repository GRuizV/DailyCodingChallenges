'''
CHALLENGES INDEX

23. Merge k Sorted Lists (LL) (DQ) (Heap) (Sorting)
53. Maximum Subarray (Array) (DQ) (DP)
105. Construct Binary Tree from Preorder and Inorder Traversal (DQ) (Tree)
108. Convert Sorted Array to Binary Search Tree (DQ) (Tree)
215. Kth Largest Element in an Array (Array) (Heap) (DQ) (Sorting)
218. The Skyline Problem (Heaps) (DQ)
240. Search a 2D Matrix II (Matrix) (DQ) (BS)
315. Count of Smaller Numbers After Self - Partially solved (Array) (DQ)
395. Longest Substring with At Least K Repeating Characters (SW) (RC) (DQ)


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


(9)
'''


'23. Merge k Sorted Lists'
# def x():

#     # Base
#     class ListNode(object):
#         def __init__(self, val=0, next=None):
#             self.val = val
#             self.next = next

#     # Input

#     # 1st Input
#     #List 1
#     one1, two1, three1 = ListNode(1), ListNode(4), ListNode(5)
#     one1.next, two1.next = two1, three1

#     #List 2
#     one2, two2, three2 = ListNode(1), ListNode(3), ListNode(4)
#     one2.next, two2.next = two2, three2

#     #List 3
#     one3, two3 = ListNode(2), ListNode(6)
#     one3.next = two3

#     # List of lists
#     li = [one1, one2, one3]

#     # My Approach

#     '''
#     Rationale:
    
#         1. Create an empty node.
#         2. Assign the node with the minimum value as next
#         3. Move that node to its next node until reaches 'None'.
#         4. When every value within the input list is None, breakout the loop and return.
#     '''

#     def mergeKLists(lists:list[ListNode]) -> ListNode:
        
#         lists = [x for x in lists if x.val != '']

#         if len(lists) == 0:
#             return ListNode('')


#         head = ListNode('')
#         curr = head
#         li = lists

#         while True:

#             if li == [None]:
#                 break

#             # Create a list of the current nodes in input that aren't None and sort them ascendingly by value
#             li = sorted([node for node in li if node != None], key = lambda x: x.val)

#             # Make the 'next_node' the next node to the curr None & move over to that node right away
#             curr.next = li[0]
#             curr = curr.next

#             # Move over to the next node of next_node
#             li[0] = li[0].next

#         return head.next

#     # Testing
#     res = mergeKLists([ListNode('')])
#     res_li = []

#     print(res)

#     'Notes: It worked'

'53. Maximum Subarray'
# def x():

#     # Input

#     # Case 1
#     nums = [-2,1,-3,4,-1,2,1,-5,4]
#     # Output: 6 / [4,-1,2,1]

#     # Case 2
#     nums = [1]
#     # Output: 1 / [1]

#     # Case 3
#     nums = [5,4,-1,7,8]
#     # Output: 23 / [5,4,-1,7,8]
    
#     '''
#     My Approach

#     Intuition: Brute forcing
#         1. Store the sum of the max array
#         2. From the max len down to len = 1, evaluate with a sliding window each sub array.
#         3. Return the max sum
#     '''
#     def maxSubArray(nums):

#         max_sum = sum(nums)

#         # Handling the special case len = 1
#         if len(nums) == 1:
#             return max_sum
        
        
#         max_element = max(nums)
#         nums_len = len(nums)-1
#         idx = 0

#         while nums_len > 1:

#             if idx + nums_len > len(nums):
#                 nums_len -= 1
#                 idx = 0
#                 continue
                
#             sub_array = nums[idx:idx+nums_len]
#             sub_array_sum = sum(sub_array)

#             if sub_array_sum > max_sum:
#                 max_sum = sub_array_sum
                
#             idx += 1

#         # Handling the case where one element is greater than any subarray sum
#         if max_element > max_sum:
#             return max_element
        
#         return max_sum


#     print(maxSubArray(nums))
    
#     'Notes: My solution worked 94,7% of the cases, the time limit was reached.'


#     '''Kadane's Algorythm (for max subarray sum)'''

#     def maxSubArray(nums):

#         max_end_here = max_so_far = nums[0]

#         for num in nums[1:]:

#             max_end_here = max(num, max_end_here + num)
#             max_so_far = max(max_so_far, max_end_here)
        
#         return max_so_far


#     print(maxSubArray(nums))

#     'Notes: Apparently it was a classic problem'

'105. Construct Binary Tree from Preorder and Inorder Traversal'
# def x():

#     # Base
#     class TreeNode(object):
#         def __init__(self, val=0, left=None, right=None):
#             self.val = val
#             self.left = left
#             self.right = right


#     # Input

#     # Case 1
#     preorder, inorder = [3,9,20,15,7],[9,3,15,20,7]
#     # Output: [3,9,20,None,None,15,7]


#     def buildTree(preorder, inorder):

#         if inorder:

#             idx = inorder.index(preorder.pop(0))
#             root = TreeNode(val = inorder[idx])
#             root.left = buildTree(preorder=preorder, inorder=inorder[:idx])
#             root.right = buildTree(preorder=preorder, inorder=inorder[idx+1:])

#             return root
        
#     'Done'

'108. Convert Sorted Array to Binary Search Tree'
# def x():

#     # Base
#     class TreeNode(object):
#         def __init__(self, val=0, left=None, right=None):
#             self.val = val
#             self.left = left
#             self.right = right

#     # Input
#     # Case 1
#     nums = [-10,-3,0,5,9]
#     # Output: [0,-3,9,-10,None,5] | [0,-10,5,None,-3,None,9]

#     # Case 2
#     nums = [1,3]
#     # Output: [3,1] | [1,None,-3]

#     '''
#     My Approach

#         Intuition:
#                 Learnt for the prior exercise, the middle node will be taken as the root.
#                 from there, it can recursively built the solution.
                
#                 base case = when len(nums) = 0
#     '''

#     def sortedArrayToBST(nums:list[int]) -> TreeNode:

#         nums_len = len(nums)

#         if nums_len:

#             idx = nums_len // 2

#             return TreeNode(val = nums[idx], left = sortedArrayToBST(nums=nums[:idx]), right = sortedArrayToBST(nums=nums[idx+1:]))

#     # Testing
#     node = sortedArrayToBST(nums=nums)
#     print(node)

#     'Done'

'''215. Kth Largest Element in an Array'''
# def x():

#     'Solution'
#     import heapq

#     def findKthLargest(self, nums: list[int], k: int) -> int:
#             heap = nums[:k]
#             heapq.heapify(heap)
            
#             for num in nums[k:]:
#                 if num > heap[0]:
#                     heapq.heappop(heap)
#                     heapq.heappush(heap, num)
            
#             return heap[0]

#     'Done'

'''218. The Skyline Problem'''
# def x():

#     '''
#     Explanation of the Code

#         Events Creation:

#             For each building, two events are created: entering ((left, -height, right)) and exiting ((right, height, 0)).
        
#         Sorting Events:

#             Events are sorted first by x-coordinate. If x-coordinates are the same, entering events are processed before exiting events. For entering events with the same x-coordinate, taller buildings are processed first.
        
#         Processing Events:

#             A max-heap (live_heap) keeps track of the current active buildings' heights. Heights are stored as negative values to use Python's min-heap as a max-heap.
#             When processing each event, heights are added to or removed from the heap as needed.
#             If the maximum height changes (top of the heap), a key point is added to the result.
        
#         This approach efficiently manages the skyline problem by leveraging sorting and a max-heap to dynamically track the highest building at each critical point.
#     '''

#     from heapq import heappush, heappop, heapify

#     def getSkyline(buildings: list[list[int]]) -> list[list[int]]:
            
#         # Create events for entering and exiting each building
#         events = []

#         for left, right, height in buildings:
#             events.append((left, -height, right))  # Entering event
#             events.append((right, height, 0))     # Exiting event
        

#         # Sort events: primarily by x coordinate, then by height
#         events.sort()
        

#         # Max-heap to store the current active buildings
#         result = []
#         live_heap = [(0, float('inf'))]  # (height, end)


#         # Process each event
#         for x, h, r in events:

#             if h < 0:  # Entering event
#                 heappush(live_heap, (h, r))

#             else:  # Exiting event
                
#                 # Remove the building height from the heap
#                 for i in range(len(live_heap)):
#                     if live_heap[i][1] == x:
#                         live_heap[i] = live_heap[-1]  # Replace with last element
#                         live_heap.pop()  # Remove last element
#                         heapify(live_heap)  # Restore heap property
#                         break
            
#             # Ensure the heap is valid
#             while live_heap[0][1] <= x:
#                 heappop(live_heap)
            
#             # Get the current maximum height
#             max_height = -live_heap[0][0]
            
#             # If the current maximum height changes, add the key point
#             if not result or result[-1][1] != max_height:
#                 result.append([x, max_height])
                    
#         return result

#     'Done'

'''240. Search a 2D Matrix II'''
# def x():

#     # Input
#     # Case 1
#     matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
#     target = 5
#     # Output: True

#     # Case 2
#     matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]]
#     target = 20
#     # Output: False


#     '''
#     My Approach

#         Intuition:

#             - Iterativelly search in the first row of the matrix if the value is in there by a belonging test
#                 - If the value is in the element, break and return True / else, pop that element from the matrix
#                 - Transpose the matrix and start over until there's no more elements in the matrix
#             - If the loop reaches the last element of the matrix, return False
#     '''

#     def search_matrix(matrix:list[list[int]], target: int) -> bool:

#         m = len(matrix)
#         n = len(matrix[0])

#         # Handle the corner case
#         if n == m == 1:
#             return target == matrix[0][0]

#         while matrix:

#             # print(matrix)

#             element = matrix.pop(0)

#             if target in element:
#                 return True
            
#             matrix = [list(x) for x in zip(*matrix)]
        
#         return False

#     # Testing
#     print(search_matrix(matrix=matrix,target=target))

#     '''Note: This approach doesn't worked because dinamically changing the data structure mess in how python checks membership'''


#     'Binary search approach'
#     def search_matrix(matrix:list[list[int]], target: int) -> bool:

#         m = len(matrix)
#         n = len(matrix[0])

#         # Handle the corner case
#         if n == m == 1:
#             return target == matrix[0][0]

#         row, col = m-1, 0   # Start the search from the bottom left corner

#         while row >= 0 and col < n:

#             element = matrix[row][col]

#             if element == target:
#                 return True
            
#             elif element > target:
#                 row -= 1
                    
#             else:
#                 col += 1
        
#         return False

#     # Testing
#     print(search_matrix(matrix=matrix,target=target))

#     'done'

'''315. Count of Smaller Numbers After Self'''
# def x():

#     # Input
#     # Case 1
#     nums = [5,2,6,1]
#     # Output: [2,1,1,0]

#     # Case 1
#     nums = [-1,-1]
#     # Output: [0,0]


#     'My Approach (Brute forcing)'
#     def count_smaller(nums: list[int]) -> list[int]:

#         # Handle corner case
#         if len(nums) == 1:
#             return [0]
        

#         # Set the min value of the group
#         min_num = min(nums)
        

#         # Initialize the result holder
#         result = []

#         for x,num in enumerate(nums):

#             # corner case: if the number is the smallest of the group or the right most one, no smaller numbers after it
#             if num == min_num or num == nums[-1]:
#                 result.append(0)
            
#             else:

#                 # Define a sublist with all elements to the right of the current one
#                 sublist = nums[x+1:]

#                 # Count how many of those are smaller than the current one
#                 count = len([x for x in sublist if x<num])

#                 # Add that result to the holder
#                 result.append(count)
                
#         return result
        
#     # Testing
#     print(count_smaller(nums=nums))

#     'Note: This approach met up to 79% o the cases'

'''395. Longest Substring with At Least K Repeating Characters'''
# def x():

#     # Input
#     # Case 1
#     s, k = "aaabb", 3
#     # Output: 3 / The longest substring is "aaa", as 'a' is repeated 3 times.

#     # Case 2
#     s, k = "ababbc", 2
#     # Output: 5 / The longest substring is "aaa", as 'a' is repeated 3 times.


#     '''
#     My approach

#         Intuition:
            
#             Brute forcing:

#                 - Import the Counter class from collections.
#                 - Initialize a max_len counter in 0 to hold the max len of a valid substring according to the requirements of k.
#                 - Starting from the len(s) down to k, check in a range, all the substrings of all those different sizes and
#                     with Counter's help check is the minimum freq is at least k,
#                         if it does: Refresh the max_len counter.
#                         if it doesn't: check the rests of the substrings.
#     '''

#     from collections import Counter

#     def longestSubstring(s: str, k: int) -> int:

#         # Initialize the max counter
#         max_len = 0

#         # Capture the len of s
#         l = len(s)

#         # Handle the corner case: len(s) < k
#         if l < k:
#             return max_len

#         # Check all possibles valid substrings
#         for i in range(k-1, l):

#             for j in range(l-i):

#                 # Create the possible valid substring
#                 substring = s[j:j+i+1]

#                 # Create a counter from the substring
#                 subs_counter = Counter(substring)

#                 # Capture the minimum freq of the caracters present
#                 subs_min_freq = min(subs_counter.values())

#                 # Update the counter only if the minimum is at least k in size
#                 max_len = len(substring) if subs_min_freq >= k else max_len


#         # Return what's un the max counter
#         return max_len

#     # Testing
#     print(longestSubstring(s=s, k=k))

#     'Note: This approach met the 87% of cases but with large input breaks. I will rethink the loop to make it go from the largest to the lowest limit, that should save some runtime.'


#     'My 2nd approach'
#     from collections import Counter

#     def longestSubstring(s: str, k: int) -> int:

#         # Capture the len of s
#         l = len(s)

#         # Handle the corner case: len(s) < k
#         if l < k:
#             return 0

#         # Check all possibles valid substrings
#         for i in range(l-1, k-2, -1):

#             if i != -1:

#                 for j in range(l-i):
                            
#                     # Create the possible valid substring
#                     substring = s[j:j+i+1]

#                     # Create a counter from the substring
#                     subs_counter = Counter(substring)

#                     # Capture the minimum freq of the caracters present
#                     subs_min_freq = min(subs_counter.values())

#                     # If the min freq found is at least k, that's the longest valid substring possible
#                     if subs_min_freq >= k:
#                         return len(substring)

#         # Return 0
#         return 0

#     # Testing
#     print(longestSubstring(s=s, k=k))

#     'Note: Unfortunately my second approach had the same performance.'


#     'Divide and Conquer approach'
#     from collections import Counter

#     def longestSubstring(s: str, k: int) -> int:

#         # Base case
#         if len(s) == 0 or len(s) < k:
#             return 0

#         # Count the frequency of eachcharacter in the string
#         counter = Counter(s)

#         # Iterate through the string and split at a character that doesn't meet the frequency requirement
#         for i, char in enumerate(s):

#             if counter[char] < k:

#                 # Split and recursively process the left and right substrings
#                 left_part = longestSubstring(s[:i], k)
#                 right_part = longestSubstring(s[i+1:], k)

#                 return max(left_part, right_part)

#         # If there's no splits, means that the entire substring is valid
#         return len(s)

#     'Done'

















