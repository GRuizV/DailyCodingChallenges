'''
CHALLENGES INDEX

4. Median of Two Sorted Arrays (Array) (BS)
34. Find First and Last Position of Element in Sorted Array (Array) (BS)
240. Search a 2D Matrix II (Matrix) (DW) (BS)

74. Search a 2D Matrix (BS) (Matrix)


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


(4)
'''



'Basic Binary Search Form'
def binary_search(low, high, condition):

    while low < high:

        mid = low + (high - low) // 2

        if condition(mid):
            high = mid  # Narrow down the range
       
        else:
            low = mid + 1  # Continue searching

    return low  # or high, depending on the case



'4. Median of Two Sorted Arrays'
# def x():

#     # Input
#     nums1 = [1,3]
#     nums2 = [2]


#     # My Approach
#     nums_total = sorted(nums1 + nums2)

#     if len(nums_total) % 2 != 0:
        
#         median_idx = len(nums_total) // 2 
#         median = float(nums_total[median_idx])
#         print(f'{median:.5f}')

#     else:
#         median_idx = len(nums_total) // 2 
#         print(nums_total[median_idx-1], nums_total[median_idx] )
#         median = (nums_total[median_idx-1] + nums_total[median_idx]) / 2
#         print(f'{median:.5f}')


#     '''
#     Note: My solution actually worked, but don't know why is not working in LeetCode.
#             Apparently, they want a solution with mergesort.
#     '''


#     # Mergesort Solution with pointers
#     def findMedianSortedArrays(nums1, nums2) -> float:
            
#             m, n = len(nums1), len(nums2)
#             p1, p2 = 0, 0
            
#             # Get the smaller value between nums1[p1] and nums2[p2].
#             def get_min():
#                 nonlocal p1, p2
#                 if p1 < m and p2 < n:
#                     if nums1[p1] < nums2[p2]:
#                         ans = nums1[p1]
#                         p1 += 1
#                     else:
#                         ans = nums2[p2]
#                         p2 += 1
#                 elif p2 == n:
#                     ans = nums1[p1]
#                     p1 += 1
#                 else:
#                     ans = nums2[p2]
#                     p2 += 1
#                 return ans
            
#             if (m + n) % 2 == 0:
#                 for _ in range((m + n) // 2 - 1):
#                     _ = get_min()
#                 return (get_min() + get_min()) / 2
#             else:
#                 for _ in range((m + n) // 2):
#                     _ = get_min()
#                 return get_min()
            
#     print(findMedianSortedArrays(nums1, nums2))

#     '''
#     Note: Honestly I feel this is kind of ego lifting.
#     '''

'34. Find First and Last Position of Element in Sorted Array'
# def x():

#     # Input

#     # case 1
#     nums = [5,7,7,8,8,10]
#     target = 8  # Expected Output: [3,4]

#     # case 2
#     nums = [5,7,7,8,8,10]
#     target = 6  # Expected Output: [-1,-1]

#     # case 3
#     nums = []
#     target = 0  # Expected Output: [-1,-1]


#     # My Approach
#     def searchRange(nums:list, target:int) -> int:
        
#         if target in nums:

#             starting_position = nums.index(target)

#             # The ending positions is calculated as of: (number of indices) - the relative position if the list is reversed
#             Ending_position = (len(nums)-1) - nums[::-1].index(target)

#             return [starting_position, Ending_position]

#         else:
#             return [-1,-1]

#     print(searchRange(nums, target))

#     'Notes: It worked!'

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




'''74. Search a 2D Matrix'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
#     target = 3
#     # Output: True

#     # Case 2
#     matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
#     target = 13
#     # Output: False

#     '''
#     My Approach

#        Apply the Binary Search idea on a matrix.
                        
#     '''

#     def searchMatrix(matrix: list[list[int]], target: int) -> bool:

#         # Handle Corner case: Target out of boundaries
#         if target < matrix[0][0] or target > matrix[-1][-1]:
#             return False
        
#         # Define the two pointers to binary search the target
#         low, high = 0, len(matrix)

#         # Start the binary search
#         while low < high:

#             # Define the mid pointer
#             mid = (low + high) // 2

#             # If the target is present in the middle element
#             if target in matrix[mid]:
#                 return True
            
#             # If the item is greater than the last item of the middle element
#             elif target > matrix[mid][-1]:

#                 # Redefine the low pointer
#                 low = mid + 1
            
#             # If the item is smaller than the first item of the middle element
#             elif target < matrix[mid][0]:

#                 # Redefine the high pointer
#                 high = mid
            
#             # If no condition is met, return False
#             else:
#                 return False

#         # If the item was not found in the loop, it means it not present in the matrix
#         return False

#     # Testing
#     print(searchMatrix(matrix=matrix, target=target))

#     '''Note: It worked right away! the results were: 6.27% in Runtime & 92.27% in Memory'''


















