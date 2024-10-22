'''
CHALLENGES INDEX

238. Product of Array Except Self (PS)

560. Subarray Sum Equals K (Array) (PS)


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


(2)
'''


'''238. Product of Array Except Self'''
# def x():

#     # Input
#     # Case 1
#     nums = [1,2,3,4]
#     # Output: [24,12,8,6]

#     # Case 2
#     nums = [-1,1,0,-3,3]
#     # Output: [0,0,9,0,0]


#     'My Approach / Brute forcing'
#     from math import prod

#     def product_except_self(nums:list[int]) -> list[int]:

#         res = []

#         for i in range(len(nums)):
#             res.append(prod(nums[:i]+nums[i+1:]))

#         return res

#     # Testing
#     print(product_except_self(nums=nums))

#     'Note: This solution suffices 75% of test cases, but resulted inefficient with large inputs'


#     'My Approach v2 - (Trying to carry the result - pseudo-prefix sum)'
#     from math import prod

#     def product_except_self(nums:list[int]) -> list[int]:

#         res = []
#         nums_prod = prod(nums)

#         for i in range(len(nums)):

#             elem = nums_prod//nums[i] if nums[i] != 0 else prod(nums[:i]+nums[i+1:])
#             res.append(elem)


#         return res

#     # Testing
#     print(product_except_self(nums=nums))

#     'Note: It worked and beated 85% in time compl. and 74% in memory compl.'


#     '''
#     Preffix-Suffix product Approach - Customary

#     Intuition:
#         - The core idea of this approach is to build an array of the carrying product of all elements from left to right (Preffix)
#             and build another more array with the sabe but from right to left (suffix).

#         - After having that by combining those products but element by element, the preffix from 0 to n-1 indexed and the suffix from n-1 to 0 indexed
#             and EXCLUDING the current index (i) in the final traversal, the 'self' element is explicitly excluded from the product.
#     '''

#     from itertools import accumulate
#     import operator

#     def product_except_self(nums:list[int]) -> list[int]:

#         res = []

#         # Populate both preffix and suffix
#         preffix = list(accumulate(nums, operator.mul))
#         suffix = list(accumulate(reversed(nums), operator.mul))[::-1]


#         # Combine the results
#         for i in range(len(preffix)):

#             if 0 < i < len(preffix)-1:
#                 res.append(preffix[i-1]*suffix[i+1])
            
#             elif i == 0:
#                 res.append(suffix[i+1])

#             else:
#                 res.append(preffix[i-1])
        
#         return res

#     # Testing
#     print(product_except_self(nums=nums))

#     'Note: Done'




'''560. Subarray Sum Equals K'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     nums = [1,1,1]
#     k = 2
#     # Output: 2

#     # Case 2
#     nums = [1,2,3]
#     k = 3
#     # Output: 2
    
#     # Custom Case
#     nums = [0,1,5,8,7,2]
#     k = 2
#     # Output: 1

#     '''
#     Solution (Prefix Sum)

#         Explanation:
            
#             1. Prefix Sum Idea:

#                 - A prefix sum is the sum of elements from the start of the array to the current position. 
#                     The key insight is that the sum of a subarray from index i to j can be computed by subtracting the prefix sum up to i-1 from the prefix sum up to j.

#                 - If current_sum is the prefix sum up to index j, and current_sum - k was seen before (at some earlier index i), it means the subarray between i+1 and j has a sum of k.
                
#             2.Hash Map Usage:

#                 - We'll use a hash map prefix_sum_count to store how many times a particular prefix sum has occurred. 
#                     This helps in determining how many subarrays ending at the current index have a sum equal to k.

#             3. Algorithm:

#                 - Traverse through the array while calculating the prefix sum. At each step, check if current_sum - k exists in the hash map, meaning we've found a subarray that sums to k.
#                 - Add the count of such subarrays to the result.
#     '''

#     def subarraySum(nums: list[int], k: int) -> int:

#         # Dictionary to store the prefix sums and their frequencies
#         prefix_sum_count = {0: 1}  # Initialize with 0 sum (one way to have a sum of 0 before starting)
#         current_sum = 0
#         count = 0
        
#         # Traverse through the array
#         for num in nums:

#             # Update the current prefix sum
#             current_sum += num
            
#             # Check if there is a previous prefix sum that makes a subarray sum to k
#             if current_sum - k in prefix_sum_count:
#                 count += prefix_sum_count[current_sum - k]
            
#             # Update the count of the current prefix sum in the map
#             if current_sum in prefix_sum_count:
#                 prefix_sum_count[current_sum] += 1
            
#             else:
#                 prefix_sum_count[current_sum] = 1
        
#         return count

#     # Testing
#     print(subarraySum(nums=nums,k=k))

#     '''Note: Done'''

































