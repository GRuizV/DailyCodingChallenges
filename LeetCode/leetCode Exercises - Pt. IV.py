'''
CHALLENGES INDEX

179. Largest Number
189. Rotate Array  (TP)
198. House Robber (DS)


(3)
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




'''xxx'''