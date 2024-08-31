'''
CHALLENGES INDEX

5. Longest Palindromic Substring (DP) (TP)
11. Container With Most Water (Array) (TP) (GRE)
15. 3Sum (Array) (TP) (Sorting)
42. Trapping Rain Water (Array) (TP) (DS) (Stack)
75. Sort Colors (Array) (TP) (Sorting)
88. Merge Sorted Array (Array) (TP) (Sorting)


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


(XX)

'''


'5. Longest Palindromic Substring'
# def x():

#     # Input
#     s = "cbbd"


#     # 1st Approach: Brute Force

#     # Creating the possible substrings from the input
#     subs = []

#     for i in range(1, len(s)+1):
        
#         for j in range((len(s)+1)-i):

#             subs.append(s[j:j+i])

#     # # validating
#     # print(subs)        

#     palindromes = sorted(filter(lambda x : True if x == x[::-1] else False, subs), key=len, reverse=True)

#     print(palindromes)

#     '''
#     Note: While the solution works, is evidently not efficient enough / Time Limit Exceeded.
#     '''

#     # 2nd Approach: Same brute force but less brute

#     max_len = 1
#     max_str = s[0]

#     for i in range(len(s)-1):

#         for j in range(i+1, len(s)):

#             sub = s[i:j+1]        

#             if (j-i)+1 > max_len and sub == sub[::-1]:

#                 max_len = (j-i)+1
#                 max_str = s[i:j+1]


#     print(max_str)

'11. Container With Most Water'
# def x():

#     # Input
#     heights = [1,8,6,2,5,4,8,3,7]


#     # My Approach
#     max_area = 0

#     for i in range(len(heights)):

#         for j in range(i+1, len(heights)):

#             height = min(heights[i], heights[j])
#             width = j-i
#             area = height * width

#             max_area = max(max_area, area)

#     print(max_area)


#     '''
#     Note:
#         While this approach works, its complexity goes up to O(n), and is required to be more efficient
#     '''


#     # Two-pointer solution

#     left = 0
#     right = len(heights)-1
#     max_area = 0

#     while left < right:

#         h = min(heights[left], heights[right])
#         width = right - left
#         area = h * width

#         max_area = max(max_area, area)

#         if heights[left] <= heights [right]:
#             left += 1
        
#         else:
#             right -= 1


#     print(max_area)

'15. 3Sum'
# def x():

#     import itertools

#     # Input
#     nums = [0,0,0]


#     # My approach

#     '''
#     Rationale:
        
#         1) Build all combinations caring for the order.
#         2) Filter down those who met sum(subset) = 0
#         3) Make sure there is no duplicates & return.

#     '''
#     comb = list(itertools.combinations(nums,3))

#     comb = [sorted(x) for x in comb if sum(x) == 0]

#     res = []

#     for i in comb:

#         if i not in res:
#             res.append(i)

#     print(res)

#     '''
#     Notes:

#         This solution actually works, but breaks when a big enough input is passed.
#     '''

#     # Two-Pointers approach solution
#     def threeSum(self, nums):
            
#             nums.sort()
#             answer = []
            
#             # if the inputs have less than 3 items
#             if len(nums) < 3:
#                 return answer
            
#             for i in range(len(nums)):

#                 # Since is a sorted input, if first element is positive, there is no way it'll sum up to 0
#                 if nums[i] > 0:
#                     break
                
#                 # Apart from the first element, if the following is the same, jump to the next iteration to avoid returning duplicates
#                 if i > 0 and nums[i] == nums[i - 1]:
#                     continue
                
#                 # Pointers setting    
#                 low, high = i + 1, len(nums) - 1

#                 while low < high:

#                     s = nums[i] + nums[low] + nums[high]

#                     if s > 0:
#                         high -= 1

#                     elif s < 0:
#                         low += 1

#                     else:

#                         answer.append([nums[i], nums[low], nums[high]])
#                         lastLowOccurrence, lastHighOccurrence = nums[low], nums[high]
                        
#                         while low < high and nums[low] == lastLowOccurrence:
#                             low += 1
                        
#                         while low < high and nums[high] == lastHighOccurrence:
#                             high -= 1
            
#             return answer

'42. Trapping Rain Water'
# def x():

#     # Input

#     # case 1
#     height = [0,1,0,2,1,0,1,3,2,1,2,1]  # Exp. Out: 6

#     # case 2
#     height = [4,2,0,3,2,5]  # Exp. Out: 9


#     'Solution'
#     def trap(height):

#         if not height:
#             return 0
        

#         left, right = 0, len(height)-1
#         left_max, right_max = 0, 0
#         result = 0

#         while left < right:

#             if height[left] < height[right]:

#                 if height[left] >= left_max:
#                     left_max = height[left]

#                 else:
#                     result += left_max - height[left]

#                 left += 1
            
#             else:

#                 if height[right] >= right_max:
#                     right_max = height[right]

#                 else:
#                     result += right_max - height[right]

#                 right -= 1
        
#         return result

#     # Testing
#     print(trap([3,0,2]))

#     'Done'

'75. Sort Colors'
# def x():

#     # Input
#     # Case 1
#     nums = [2,0,2,1,1,0]
#     # Output: [0,0,1,1,2,2]

#     # Case 2
#     nums = [2,0,1]
#     # Output: [0,1,2]

#     '''
#     My approach

#         Intuition:
#             Since the solution requires the sorting be in place, 
#             perhaps Bubblesort would do the trick.
#     '''

#     def sortColors(nums:list[int]) -> list[int]:

#         swapped = True       

#         while swapped != False:

#             swapped = False
#             i = 0

#             while True:

#                 if i == len(nums)-1:
#                     break

#                 if nums[i] > nums[i+1]:
#                     nums[i], nums[i+1] = nums[i+1], nums[i]
#                     swapped = True

#                 i += 1

#     # Testing
#     sortColors(nums)
#     print(nums)

#     'Notes: Done!'

'88. Merge Sorted Array'
# def x():

#     # Input
#     # Case 1
#     nums1 = [1,2,3,0,0,0]
#     m = 3
#     nums2 = [2,5,6]
#     n = 3
#     # Output: [1,2,2,3,5,6]

#     # Case 2
#     nums1 = [1]
#     m = 1
#     nums2 = []
#     n = 0
#     # Output: [1]

#     # Case 3
#     nums1 = [0]
#     m = 0
#     nums2 = [1]
#     n = 1
#     # Output: [1]

#     # Custom case
#     nums1 = [0,2,0,0,0,0,0]
#     m = 2
#     nums2 = [-1,-1,2,5,6]
#     n = 5
#     # Output: [1]

#     # Custom case
#     nums1 = [-1,1,0,0,0,0,0,0]
#     m = 2
#     nums2 = [-1,0,1,1,2,3]
#     n = 6
#     # Output: [1]


#     'Solution'
#     def merge(nums1, m, nums2, n):

#         if m == 0:
#             for i in range(n):
#                 nums1[i] = nums2[i]

#         elif n != 0:

#             m = n = 0

#             while n < len(nums2):

#                 if nums2[n] < nums1[m]:

#                     nums1[:m], nums1[m+1:] = nums1[:m] + [nums2[n]], nums1[m:-1]

#                     n += 1
#                     m += 1
                
#                 else:

#                     if all([x==0 for x in nums1[m:]]):
#                         nums1[m] = nums2[n]
#                         n += 1
                        
#                     m += 1

#     # Testing
#     merge(nums1,m,nums2,n)
#     print(nums1)























