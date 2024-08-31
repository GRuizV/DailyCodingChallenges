'''
CHALLENGES INDEX

*Arrays
1. Two Sum (Array) (Hash Table)
4. Median of Two Sorted Arrays (Array)
11. Container With Most Water (Array)
15. 3Sum (Array)
34. Find First and Last Position of Element in Sorted Array (Array)
36. Valid Sudoku (Array) (Hash Table) (Matrix)
42. Trapping Rain Water (Array)
46. Permutations (Array)
48. Rotate Image (Array) (Matrix)
49. Group Anagrams (Array) (Hash Table) (Sorting)
53. Maximum Subarray (Array) (DQ) (DP)
55. Jump Game (Array) (DP) (GRE)
75. Sort Colors (Array) (TP) (Sorting)
78. Subsets (Array) (BT)
88. Merge Sorted Array (Array) (TP) (Sorting)


*Hash Tables
3. Longest Substring Without Repeating Characters (Hash Table) (SW)
13. Roman to Integer (Hash Table)
17. Letter Combinations of a Phone Number (Hash Table) (BT)
73. Set Matrix Zeroes (Matrix) (Hash Table)
76. Minimum Window Substring (Hash Table) (SW)


*Matrices
54. Spiral Matrix (Matrix)  
79. Word Search (Matrix) (BT)


*Others
7. Reverse Integer (Others)
8. String to Integer (atoi) (Others)
14. Longest Common Prefix (Others)
29. Divide Two Integers (Others)
38. Count and Say (Others)
66. Plus One (Others)
69. Sqrt(x) (Others)


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

'### ARRAYS ###'

'1. Two Sum'
# def x():
#     # 1st approach
#     nums = [3,2,4]
#     target = 6
#     result = []

#     for i in range(len(nums)):

#         for j in range(i+1, len(nums)):

#             if nums[i] + nums[j] == target:
#                 result = [i, j]
#                 break
            
#         if result:
#             break
        
#     print(result)


#     # 2nd approach
#     nums = [3,2,4]
#     target = 6
#     hashmap = {v:k for k,v in enumerate(nums)}
#     result = []

#     for i in range(len(nums)):

#         comp = target - nums[i]

#         if comp in hashmap and i != hashmap[comp]:
#             result = [i, hashmap[comp]]
#             break
    
    
#     print(result)


#     # 3rd approach
#     nums = [3,2,4]
#     target = 6
#     hashmap = {}
#     result = []

#     for i in range(len(nums)):

#         comp = target - nums[i]

#         if comp in hashmap:
#             result = [i, hashmap[comp]]
#             break

#         hashmap[nums[i]] = i
        
#     print(result)

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

'36. Valid Sudoku'
# def x():

#     # Input

#     # Case 1
#     board = [
#     ["5","3",".",".","7",".",".",".","."]
#     ,["6",".",".","1","9","5",".",".","."]
#     ,[".","9","8",".",".",".",".","6","."]
#     ,["8",".",".",".","6",".",".",".","3"]
#     ,["4",".",".","8",".","3",".",".","1"]
#     ,["7",".",".",".","2",".",".",".","6"]
#     ,[".","6",".",".",".",".","2","8","."]
#     ,[".",".",".","4","1","9",".",".","5"]
#     ,[".",".",".",".","8",".",".","7","9"]
#     ]

#     # Case 2
#     board = [
#     ["8","3",".",".","7",".",".",".","."]
#     ,["6",".",".","1","9","5",".",".","."]
#     ,[".","9","8",".",".",".",".","6","."]
#     ,["8",".",".",".","6",".",".",".","3"]
#     ,["4",".",".","8",".","3",".",".","1"]
#     ,["7",".",".",".","2",".",".",".","6"]
#     ,[".","6",".",".",".",".","2","8","."]
#     ,[".",".",".","4","1","9",".",".","5"]
#     ,[".",".",".",".","8",".",".","7","9"]
#     ]


#     # My Approach
#     '''
#     Rationale:
#         1. Pull out all the columns, rows and sub-boxes to be evaluated.
#         2. Filter down empty colums, rows and sub-boxes.
#         3. Cast set on each element on the 3 groups and 
#             if one of them have less items than before the casting, return False. Otherwise, return True
#     '''

#     def isValidSudoku(board: list[list[str]]) -> bool:

#         rows = board
#         columns = [list(x) for x in zip(*board)]


#         # Bulding the sub-boxes directly into the list
#             # Did it this way to save time complexity.
                
#         sub_boxes = [
#             [board[0][0:3],board[1][0:3],board[2][0:3]],
#             [board[0][3:6],board[1][3:6],board[2][3:6]],
#             [board[0][6:9],board[1][6:9],board[2][6:9]],
#             [board[3][0:3],board[4][0:3],board[5][0:3]],
#             [board[3][3:6],board[4][3:6],board[5][3:6]],
#             [board[3][6:9],board[4][6:9],board[5][6:9]],
#             [board[6][0:3],board[7][0:3],board[8][0:3]],
#             [board[6][3:6],board[7][3:6],board[8][3:6]],
#             [board[6][6:9],board[7][6:9],board[8][6:9]],
#         ]


#         # Validating rows
#         for row in rows:

#             row_wo_dot = [num for num in row if num != '.']

#             if len(row_wo_dot) != len(set(row_wo_dot)):
#                 return False


#         # Validating columns
#         for col in columns:

#             col_wo_dot = [num for num in col if num != '.']

#             if len(col_wo_dot) != len(set(col_wo_dot)):
#                 return False


#         # Validating Sub-boxes
#         for subb in sub_boxes:

#             plain_subb = [num for li in subb for num in li if num != '.']

#             if len(plain_subb) != len(set(plain_subb)):
#                 return False


#         return True


#     print(isValidSudoku(board))

#     'Notes: It works perfectly, but could be less verbose'


#     # Another Approach
#     import collections

#     def isValidSudoku(self, board):

#         rows = collections.defaultdict(set)
#         cols = collections.defaultdict(set)
#         subsquares = collections.defaultdict(set)

#         for r in range(9):

#             for c in range(9):

#                 if(board[r][c] == "."):
#                     continue

#                 if board[r][c] in rows[r] or board[r][c] in cols[c] or board[r][c] in subsquares[(r//3, c//3)]:
#                     return False
                
#                 rows[r].add(board[r][c])
#                 cols[c].add(board[r][c])
#                 subsquares[(r//3,c//3)].add(board[r][c])

#         return True

#     '''
#     Notes: 
#         This solution was much more elegant. And essentially the difference lays in this solution could be more scalable 
#         since it builds the data holder while iterating.
#     '''

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

'48. Rotate Image'
# def x():

#     # Input

#     # Case 1
#     matrix = [
#         [1,2,3],
#         [4,5,6],
#         [7,8,9]
#         ]
#     # Exp. Out: [[7,4,1],[8,5,2],[9,6,3]]

#     # Case 2
#     matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
#     # Exp. Out: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]


#     'My approach'
#     def rotate(matrix: list[list[int]]):

#         n = len(matrix)

#         for i in range(n):

#             rot_row = []

#             for j in range(n):  # Since is given that is an squared matrix
#                 rot_row.insert(0, matrix[j][i])

#             matrix.append(rot_row)
    
#         for i in range(n):
#             matrix.pop(0)

#     # Testing
#     rotate(matrix)
#     print(matrix)

#     'Notes: It worked, but seems a little unorthodox'


#     'Another Approach'
#     def rotate(matrix):

#         # reverse
#         l = 0
#         r = len(matrix) -1

#         while l < r:
#             matrix[l], matrix[r] = matrix[r], matrix[l]
#             l += 1
#             r -= 1

#         x = ''
#         matrix = matrix[::-1]
#         x=0

#         # transpose 
#         for i in range(len(matrix)):
#             for j in range(i):
#                 matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


#     # Testing
#     rotate(matrix)
#     print(matrix)

#     'Notes: This one looks much more like a canon type solution'

'49. Group Anagrams'
# def x():

#     # Input

#     # Case 1
#     strs = ["eat","tea","tan","ate","nat","bat"]
#     #Exp. Out: [["bat"],["nat","tan"],["ate","eat","tea"]]

#     # Case 2
#     strs = [""]
#     #Exp. Out: [[""]]

#     # Case 3
#     strs = ["a"]
#     # Exp. Out: [["a"]]

#     # Custom Case
#     strs = ["ddddddddddg","dgggggggggg"]
#     # Expected: [["dgggggggggg"],["ddddddddddg"]]



#     'My Approach'

#     '''
#     Intuition:
#         1. Take the first element of the input and make a list with all element that contains the same characters
#         2. Erase the taken elements from the input.
#         3. Reiterate steps 1 & 2 until the input is exhausted

#     '''

#     def groupAnagrams(strs:list):
        
#         if len(strs) == 1:
#             return[strs]

#         # Auxiliary anagram checker
#         def is_anagram(ref:list, string:list):

#             if len(ref) != len(string):
#                 return False

#             for char in ref:
                
#                 if ref.count(char) != string.count(char):   
#                     return False

#             return True
        
#         # Creating Flag to manage repetitions
#         strs = [[word, False] for word in strs]


#         result = []

#         for word in strs:
                
#             if word[1] == False:

#                 anagrams = []
#                 anagrams.append(word[0])            
#                 word[1] = True

#                 for rest in strs:

#                     if rest[1] == False:

#                         if is_anagram(word[0], rest[0]):
#                             anagrams.append(rest[0])
#                             rest[1] = True
            
#                 result.append(anagrams)

#         return result
    
#     # Testing
#     print(groupAnagrams(strs))

#     '''
#     Notes: 
#         It passed 72/126 cases, the case below broke the code: 
#             strs = ["ddddddddddg","dgggggggggg"] / Output: [["ddddddddddg","dgggggggggg"]], Expected: [["dgggggggggg"],["ddddddddddg"]]

#         After the fixture, it works but beat no one in efficiency
#     '''


#     'Another Approach'
#     def groupAnagrams(strs):
        
#         freq = {}

#         for word in strs:

#             newWord = ''.join(sorted(word))

#             if newWord not in freq:
#                 freq[newWord] = []
            
#             freq[newWord].append(word)

#         return list(freq.values())

#     # Testing
#     print(groupAnagrams(strs))

#     '''
#     Notes: Absolutely more elegant solution
#     '''

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

'55. Jump Game'
# def x():

#     # Input
#     # Case 1
#     nums = [2,3,1,1,4]
#     # Output: True

#     # Case 2
#     nums = [3,2,1,0,4]
#     # Output: False

#     # Custom Case
#     nums = [2,0,0]
#     # Output: 

#     '''
#     My Approach

#         Intuition - Brute Force:

#             - I will check item by item to determine if the end of the list is reachable

#     '''
#     def canJump(nums:list[int]) -> bool:

#         # Corner case: nums.lenght = 1 / nums[0] = 0
#         if len(nums) == 1 and nums[0] == 0:
#             return True
        
#         idx = 0

#         while True:

#             idx += nums[idx]

#             if idx >= len(nums)-1:
#                 return True
            
#             if nums[idx] == 0 and idx < len(nums)-1:
#                 return False
            
#     # Testing
#     print(canJump(nums))

#     'Notes: This solution suffice 91,2% of the case'


#     'Backtrack Approach'
#     def canJump(nums: list[int]) -> bool:

#         if len(nums)==1:
#             return True  

#         #Start at num[-2] since nums[-1] is given
#         backtrack_index = len(nums)-2 
#         #At nums[-2] we only need to jump 1 to get to nums[-1]
#         jump =1  

#         while backtrack_index>0:
#             #We can get to the nearest lily pad
#             if nums[backtrack_index]>=jump: 
#                 #now we have a new nearest lily pad
#                 jump=1 
#             else:
#                 #Else the jump is one bigger than before
#                 jump+=1 
#             backtrack_index-=1
        
#         #Now that we know the nearest jump to nums[0], we can finish
#         if jump <=nums[0]: 
#             return True
#         else:
#             return False 

#     'Notes: Right now I am not that interested in learning bactktracking, that will be for later'

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








'### HASH TABLE ###'

'3. Longest Substring Without Repeating Characters'
# def x():
#     s = "abcabcbb"


#     # My solution
#     substrings = []

#     i = 0

#     while i < len(s):

#         sub = str()

#         for char in s[i:]:

#             if char in sub:
#                 substrings.append(sub)
#                 break

#             sub += char
        
#         if sub not in substrings:
#             substrings.append(sub)

#         i += 1

#     # print(substrings)

#     max_sub = max(substrings, key = len) if substrings else 0

#     # print(max_sub)

#     print(max_sub, len(max_sub))


#     # Another more efficient solution

#     def lengthOfLongestSubstring(s: str) -> int:
            
#             n = len(s)
#             maxLength = 0
#             charMap = {}
#             left = 0
            
#             for right in range(n):

#                 if s[right] not in charMap or charMap[s[right]] < left:
#                     charMap[s[right]] = right
#                     maxLength = max(maxLength, right - left + 1)

#                 else:
#                     left = charMap[s[right]] + 1
#                     charMap[s[right]] = right
            
#             return maxLength


#     lengthOfLongestSubstring(s)

'13. Roman to Integer'
# def x():

#     '''
#     Substraction exceptions:
#         - I can be placed before V (5) and X (10) to make 4 and 9. 
#         - X can be placed before L (50) and C (100) to make 40 and 90. 
#         - C can be placed before D (500) and M (1000) to make 400 and 900.
#     '''

#     # Input
#     s = 'MCMXCIV'


#     # My approach
#     res = 0
#     rom_to_int_dic = {'I': 1, 'IV': 4, 'V': 5, 'IX': 9, 'X': 10, 'XL': 40, 'L': 50, 'XC': 90, 'C': 100, 'CD': 400, 'D': 500, 'CM': 900, 'M': 1000, }


#     #Substraction Exceptions
#     if 'IV' in s:
#         res += rom_to_int_dic['IV']
#         s = s.replace('IV','')

#     if 'IX' in s:
#         res += rom_to_int_dic['IX']
#         s = s.replace('IX','')

#     if 'XL' in s:
#         res += rom_to_int_dic['XL']
#         s = s.replace('XL','')

#     if 'XC' in s:
#         res += rom_to_int_dic['XC']
#         s = s.replace('XC','')

#     if 'CD' in s:
#         res += rom_to_int_dic['CD']
#         s = s.replace('CD','')

#     if 'CM' in s:
#         res += rom_to_int_dic['CM']
#         s = s.replace('CM','')

#     # Dealing with the Remaining Number
#     if s:
#         for chr in s:
#             res += rom_to_int_dic[chr]

#     else:
#         print(res)


#     print(res)

#     '''
#     Note: This version works, but there is a more concise way
#     '''

#     s = 'MCMXCIV'

#     # ChatGPT's Approach
#     roman_dict = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
#     total = 0
#     prev_value = 0

#     for char in s[::-1]:    #Reverse to simplify the process
        
#         curr_value = roman_dict[char]

#         if curr_value < prev_value:
#             total -= curr_value
        
#         else:
#             total += curr_value
#             prev_value = curr_value

#     print(total)

'17. Letter Combinations of a Phone Number'
# def x():

#     # Input
#     s = '23'

#     '''
#     My Approach

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

'73. Set Matrix Zeroes'
# def x():

#     # Input
#     # Case 1
#     matrix = [[1,1,1],[1,0,1],[1,1,1]]
#     # Output: [[1,0,1],[0,0,0],[1,0,1]]

#     # Case 2
#     matrix = [[0,1,2,0],[3,4,5,2],[1,3,1,5]]
#     # Output: [[0,0,0,0],[0,4,5,0],[0,3,1,0]]

#     # Custom Case
#     matrix = [...]
#     # Output: [...]

#     '''
#     My Approach

#         Intuition:
#             - Locate the indexes of every 0 present
#             - Try to overwrite the values for the row and the column of each occurrence
#             - Look up if the col and row are already 0 to optimize
#     '''

#     def setZeroes(matrix: list[list[int]]) -> list[list[int]]:

#         m, n = len(matrix), len(matrix[0])
#         occurrences = []

#         for i, row in enumerate(matrix):

#             for j, col in enumerate(row):

#                 if 0 not in row:
#                     continue
                
#                 if col == 0:
#                     occurrences.append((i,j))

#         for pair in occurrences:

#             matrix[pair[0]] = [0] * n

#             for row in range(m):
#                 matrix[row][pair[1]] = 0
        
#         return matrix

#     # Testing
#     for i in setZeroes(matrix):
#         print(i)

#     '''
#     Notes: It actually passed! :D
#     '''

'76. Minimum Window Substring'
# def x():

#     # Input
#     # Case 1
#     s, t = 'ADOBECODEBANC', 'ABC'
#     # Output: "BANC"

#     # Case 2
#     s, t = 'a', 'a'
#     # Output: "a"

#     # Case 3
#     s, t = 'a', 'aa'
#     # Output: "abbbbbcdd"

#     # Custom case
#     s, t = 'aaaaaaaaaaaabbbbbcdd', 'abcdd'
#     # Output: "abbbbbcdd"


#     'My approach'
#     def minWindow(s:str, t:str) -> str:

#         if len(t) > len(s):
#             return ''
        
#         if t == s:
#             return t
        

#         for i in range(len(t), len(s) + 1):

#             for j in range((len(s)-i) + 1):
                
#                 if all([char in s[j:j+i] for char in t]):
#                     return s[j:j+i]
                
#         return ''

#     'Notes: This solution works up to 57%'


#     'With an improvement'
#     def minWindow(s:str, t:str) -> str:

#         from collections import Counter

#         if len(t) > len(s):
#             return ''
        
#         if t == s:
#             return t
        
#         count_t = Counter(t).items()

#         for i in range(len(t), len(s) + 1):

#             for j in range((len(s)-i) + 1):
                
#                 subs = s[j:j+i]
#                 count_subs = Counter(subs)

#                 if all( (x[0] in count_subs.keys() and x[1] <= count_subs[x[0]]) for x in count_t):
#                     return s[j:j+i]
                
#         return ''

#     'Notes: This solution works up to 93% and hit the time limit'


#     'Another solution'
#     def minWindow(s, t):    

#         if not s or not t:
#             return ""


#         from collections import defaultdict

#         dictT = defaultdict(int)
#         for c in t:
#             dictT[c] += 1

#         required = len(dictT)
#         l, r = 0, 0
#         formed = 0

#         windowCounts = defaultdict(int)
#         ans = [-1, 0, 0]

#         while r < len(s):
#             c = s[r]
#             windowCounts[c] += 1

#             if c in dictT and windowCounts[c] == dictT[c]:
#                 formed += 1

#             while l <= r and formed == required:
#                 c = s[l]

#                 if ans[0] == -1 or r - l + 1 < ans[0]:
#                     ans[0] = r - l + 1
#                     ans[1] = l
#                     ans[2] = r

#                 windowCounts[c] -= 1
#                 if c in dictT and windowCounts[c] < dictT[c]:
#                     formed -= 1

#                 l += 1

#             r += 1

#         return "" if ans[0] == -1 else s[ans[1]:ans[2] + 1]
            
#     # Testing
#     print(minWindow(s,t))








'### MATRICES ###'

'54. Spiral Matrix'
# def x():

#     # Input
#     # Case 1
#     matrix = [[1,2,3],[4,5,6],[7,8,9]]
#     # Output: [1,2,3,6,9,8,7,4,5]

#     # Case 2
#     matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
#     # Output: [1,2,3,4,8,12,11,10,9,5,6,7]

#     # Custom Case  / m = 1
#     matrix = [[2,1,3]]
#     # Output: [2,1,3]

#     # Custom Case / n = 1
#     matrix = [[2],[1],[3],[5],[4]]
#     # Output: [2,1,3,5,4]

#     # Custom Case
#     matrix = [[1,2],[3,4]]
#     # Output: [1,2,4,3]

#     # Custom Case
#     matrix = [[2,3,4],[5,6,7],[8,9,10],[11,12,13],[14,15,16]]
#     # Output: [2,3,4,7,10,13,16,15,14,11,8,5,6,9,12]

#     '''
#     My approach

#         Intuition:
#             1. Handle Special Cases: m = 1 / n = 1 

#             2. Make a while loop that runs until the input has no elements:
#                 a. Take all the subelements from the first element and append them individually to the result.
#                 b. Make a for loop and take the last subelement of each element and append them individually to the result
#                 c. Take all the subelements from the last element and append them in a reverse order individually to the result.
#                 d. Make a for loop and take the first subelement of each element and append them in a reverse order individually to the result.

#             3. Return the result
#     '''

#     def spiralOrder(matrix):

#         if len(matrix) == 1:
#             return matrix[0]
        
#         if len(matrix[0]) == 1:
#             return [num for vec in matrix for num in vec]
        

#         result = []

#         while len(matrix) != 0:

#             first_element = matrix.pop(0)
#             result += first_element

#             if len(matrix) == 0:
#                 break

#             second_element = []
#             for elem in matrix:
#                 second_element.append(elem.pop(-1))
#             result += second_element

#             third_element = matrix.pop(-1)
#             result += reversed(third_element)
            
#             if len(matrix) > 0 and len(matrix[0]) == 0:
#                 break

#             fourth_element = []
#             for elem in matrix:
#                 fourth_element.append(elem.pop(0))
#             result += reversed(fourth_element)

#         return result

#     print(spiralOrder(matrix))


#     'Notes: it works up to 76% of the cases, but from here seems more like patching something that could be better designed'


#     'Another Approach'

#     def spiralOrder(matrix):
            
#             result = []

#             while matrix:
#                 result += matrix.pop(0) # 1

#                 if matrix and matrix[0]: # 2 
#                     for line in matrix:
#                         result.append(line.pop())

#                 if matrix: # 3
#                     result += matrix.pop()[::-1]

#                 if matrix and matrix[0]: # 4

#                     for line in matrix[::-1]:
#                         result.append(line.pop(0))

#             return result

#     'Notes: Same logic, better executed'

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








'### OTHERS ###'

'7. Reverse Integer'
# def x():

#     # Input
#     x = -15

#     # My Solution
#     raw_res = str(x)[::-1]

#     if '-' in raw_res:
#         res = int(raw_res[:-1])
#         res = int('-'+str(res))

#     else:
#         res = int(raw_res)


#     min_32bit = -2**31
#     max_32bit = 2**31-1

#     if res > max_32bit or res < min_32bit:
#         print(0)

#     else:
#         print(res)

'8. String to Integer (atoi)'
# def x():

#     # Input
#     s = "   -43.25"


#     # My approach

#     # Cleaning leading blank spaces
#     s = s.strip()

#     # In case of a sign present
#     sign = None

#     if s[0] == '-' or s[0] == '+':    
#         sign = s[0]
#         s = s[1:]


#     num = ''

#     # Reviewing each valid character
#     for char in s:    
#         if char not in '0123456789.':
#             break    
#         num += char


#     decimals = None

#     # In case of a decimals
#     if '.' in num:
#         decimals = num[num.find('.')+1:] #35
#         num = num[:num.find('.')]   #42
#         decimal_break = 5 * 10**(len(decimals)-1)

#         decimals = int(decimals)
        
#         #in case no number befor '.'
#         if not num:
#             num = 0
#         else:
#             num = int(num)
        
#         if decimals >= decimal_break:
#             num += 1

#     elif num:
#         num = int(num)



#     # In case is negative
#     if sign == '-':
#         num = int('-'+str(num))


#     max_32bit = 2**31-1
#     min_32bit = -2**31


#     #Outputting the result
#     if not num:
#         print(0)

#     else:
    
#         if num < min_32bit:
#             print(min_32bit)

#         elif num > max_32bit:
#             print(max_32bit)
        
#         else:      
#             print(num)

#     ''' 
#     Note:
#         It left cases unhandled. I also don't have the time to keep building the solution.
#     '''

#     # ChatGPT approach

#     def atoi(s: str) -> int:
#         s = s.strip()  # Remove leading and trailing whitespace
#         if not s:
#             return 0
        
#         sign = 1
#         i = 0
        
#         # Check for sign
#         if s[i] in ['+', '-']:
#             if s[i] == '-':
#                 sign = -1
#             i += 1
        
#         # Iterate through characters and build the number
#         num = 0
#         while i < len(s) and s[i].isdigit():
#             num = num * 10 + int(s[i])
#             i += 1
        
#         # Apply sign and handle overflow
#         num *= sign
#         num = max(-2**31, min(2**31 - 1, num))
        
#         return num

#     print(atoi(s))

'14. Longest Common Prefix'
# def x():

#     '''
#     Approach:

#         1. Order the array alphabetically 
#             & separate in different lists the words starting with each letter.

#         2. Order each array with the longest word upfront.

#         3. Build a dict with *Preffix as key and *Count as value.
#             *Count: will be how many words start with the first letter of the first word, the first two letters of the first word, 
#             and so on until exhauting the first (longest) word
#             *Preffix: the actual first, two first and so on substrings.

#         4. Merge all the resulting dict, order them descendingly, and return the first value if the count > 2, else: return an empty string.

#     '''

#     # Input

#     #   Custom input for testing
#     strs = ["flower", "flow", "flight", "dog", "racecar", "door", "fleet", "car", "racer"]

#     #   Real input
#     strs = ["a"]




#     # My approach

#     def longestCommonPrefix(strs):

#         strs = sorted(strs, reverse=True)

#         # Here will be stored each list
#         lists = {}

#         for str in strs:

#             first_letter = str[0]

#             if first_letter in lists:
#                 lists[first_letter].append(str)
            
#             else:
#                 lists[first_letter] = [str]


#         # Converting the dict into a list to facilitate the logic
#         groups = list(lists.values())

#         # Ordering each sublist by len
#         groups = list(map(lambda x: sorted(x, key=len, reverse=True), groups))

#         # Here will be the counting and the prefixes
#         results = dict()


#         for group in groups:

#             for i in range(1, len(group[0])):
                    
#                 prefix = ''
#                 count = 0

#                 for j in range(len(group)):

#                     if group[0][:i] in group[j]:
#                         count += 1
                    
#                 if count > 1:

#                     prefix = group[0][:i]
#                     results[prefix] = count


#         results = sorted(results.items(), key = lambda x: (x[1], x[0]), reverse=True)

#         # print(results)


#         if results:
#             return results[0][0]

#         else:
#             return ''
        

#     print(longestCommonPrefix(strs))

#     '''
#     Note:
#         My solution appears to be functional but is not working as expected with unexpected input.
#     '''

#     # ChatGPT's approach

#     def longestCommonPrefix(strs):

#         if not strs:
#             return ''

#         strs.sort()
#         first = strs[0]
#         last = strs[-1]

#         prefix = ''

#         for i in range(len(first)):

#             if i < len(last) and first[i] == last[i]:
                
#                 prefix += first[i]
            
#             else:
#                 break
        
#         return prefix


#     print(longestCommonPrefix(strs))

#     '''
#     Conclusion: The difference between my implementation and this is that the problem didn't state that the prefix must be present in all the strings, I assumed it wasn't going to be.
#     '''

'29. Divide Two Integers'
# def x():

#     # Input

#     # Case 1
#     dividend = 10
#     divisor = 3

#     # Case 2
#     dividend = 7
#     divisor = -3

#     # My approach

#     '''
#     Rationale:
#         1. Count how many times the divisor could be substracted from the dividend before reaching something smalle than the divisor
#         2. if only one between dividend and the divisor is less than 0, the result would return a negative number 
#     '''


#     def divide(dividend, divisor):
        
#         # case where 0 is divided by something
#         if dividend == 0:
#             return 0
        

#         # setting variables to operate
#         count = 0
#         div = abs(divisor)
#         dvnd = abs(dividend)


#         # case where the dividend is 1
#         if div == 1 and dvnd != 0:

#             if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0):
#                 return -dvnd
            
#             else:
#                 return dvnd
        

#         # case where the absolute divisor is greater than the dividend
#         if div > dvnd:
#             return 0
        
#         # case where both are the same number
#         if div == dvnd:
                
#             if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0):
#                 return -1
            
#             else:
#                 return 1
        
#         # case where is possible to divide iteratively
#         while dvnd >= div:

#             dvnd -= div
#             count += 1
        
#         # In case any is negative, the result will also be negative
#         if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0):
#             return -count

#         # Otherwise, just return
#         return count


#     print(divide(dividend, divisor))

#     'Notes: My solution actually worked, theres nitpicking cases where it wont, but still '


#     # Another Approach

#     def divide(dividend, divisor):

#         if (dividend == -2147483648 and divisor == -1): return 2147483647
                
#         a, b, res = abs(dividend), abs(divisor), 0

#         for x in range(32)[::-1]:

#             if (a >> x) - b >= 0:
#                 res += 1 << x
#                 a -= b << x
        
#         return res if (dividend > 0) == (divisor > 0) else -res

#     'Notes: This challenge is solved with bitwise operations '

'38. Count and Say'
# def x():

#     # Input

#     # Case 1
#     n = 1   # Exp. Out: "1" (Base Case)

#     # Case 2
#     n = 4   # Exp. Out: "1211" (Base Case)


#     'My Approach - Iterative suboptimal solution' 
#     def countAndSay(n):

#         if n == 1:
#             return '1'
            
#         res = '1'

#         for _ in range(1, n):
        
#             pairs = []
#             count = 0
#             char = res[0]

#             for i in range(len(res)+1):

#                 if i == len(res):
#                     pairs.append(str(count)+char)

#                 elif res[i] == char:
#                     count += 1

#                 else:       
#                     pairs.append(str(count)+char)
#                     char = res[i]
#                     count = 1

#             res = ''.join(pairs)
        
#         return res

#     # Testing
#     print(countAndSay(6))

#     'Notes: It works'

#     'Recursive Approach'
#     def countAndSay(n):
#         if n == 1:
#             return '1'
#         return aux_countAndSay(countAndSay(n - 1))

#     def aux_countAndSay(s):
    
#         if not s:
#             return ''
        
#         result = []
#         count = 1

#         for i in range(1, len(s)):

#             if s[i] == s[i - 1]:
#                 count += 1

#             else:
#                 result.append(str(count) + s[i - 1])
#                 count = 1

#         result.append(str(count) + s[-1])

#         return ''.join(result)

#     # Testing
#     print(countAndSay(6))

#     'Done'

'66. Plus One'
# def x():

#     # Input
#     # Case 1
#     digits = [1,2,3]
#     # Output: [1,2,4]

#     # Case 2
#     digits = [4,3,2,1]
#     # Output: [4,3,2,2]

#     # Case 3
#     digits = [9]
#     # Output: [1,0]

#     # Custom Case
#     digits = [9,9,9]
#     # Output: [1,0,0,0]

#     '''
#     My Approach
#         Intuition:
#             - The case is simple, the catch is to handle the case "[9,9,9]"
#     '''

#     def plusOne(digits: list[int]) -> list[int]:

#         idx = -1

#         while abs(idx) <= len(digits):
            
#             if abs(idx) == len(digits) and digits[idx] == 9:

#                 digits[idx] = 1
#                 digits.append(0)
#                 break

#             if digits[idx] != 9:

#                 digits[idx] += 1
#                 break

#             digits[idx] = 0
#             idx -= 1

#         return digits

#     # Testing
#     print(plusOne(digits=digits))

#     '''
#     Notes: 

#         While this code works, there was an even cleverer approach - To convert the digits into a int, add 1 and return as a list of ints
#         this way, is avoided the handling of cases
#     '''


#     'A different Approach'
#     def plusOne(digits: list[int]) -> list[int]:

#         number = int(''.join([str(x) for x in digits]))
#         number += 1
        
#         return [int(x) for x in str(number)]

#     # Testing
#     print(plusOne(digits=digits))

'69. Sqrt(x)'
# def x():

#     # Input
#     # Case 1
#     x = 4
#     # Output: 2

#     # Case 2
#     x = 8
#     # Output: 2

#     # Custom Case
#     x = 399
#     # Output: ..

#     'My Approach'

#     limit = 46341

#     # Auxiliary Eratosthenes' sieve function
#     def primes(cap):  

#         primes = []
#         not_primes = []

#         for i in range(2, cap+1):

#             if i not in not_primes:
#                 primes.append(i)
#                 not_primes.extend([x for x in range(i*i, cap+1, i)])

#         return primes

#     def mySqrt(x:int) -> int:

#         #Setting a limit for calculating primes
#         limit = x//2

#         prime_nums = primes(limit)

#         squares = list(map(lambda x: x*x, prime_nums))


#         #proning in the squares the correct range to make a range to evaluate
#         root_range = []
#         for idx, v in enumerate(squares):

#             if x <= v:
#                 root_range = [prime_nums[idx-1], prime_nums[idx]]
#                 break

#         #Calculating manually the square of each root in range to select the floor-root for the value
#         for root in range(root_range[0], root_range[1]+1):
            
#             if root*root >= x:
#                 return result
            
#             result = root

#     # Testing
#     print(mySqrt(x))

#     'Notes: This approach was too complicated and actually not as efficient. Apparently with the notion of binary search is easier to solve'


#     'Binary Search Approach'
#     def mySqrt(x):

#         left = 0
#         right = x

#         while left <= right:

#             mid = (left + right)//2

#             if mid*mid < x:
#                 left = mid + 1

#             elif mid*mid > x: 
#                 right = mid -1

#             else:
#                 return mid
        
#         return right

#     # Testing
#     print(mySqrt(x))


























