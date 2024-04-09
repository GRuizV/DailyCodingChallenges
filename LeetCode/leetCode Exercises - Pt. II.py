'20. Valid Parentheses'

# input / Case - expected result
# s = '()'    # True
# s = '()[]{}'    # True
# s = '(]'    # False
# s = '([({[]{}}())])'    # True
# s = '([({[)]{}}())])'    # False
# s = '))'    # False
# s = '(('    # False



# My approach

# def isValid(s):

#     stack = list(s)
#     temp = []
#     dic = {'(': ')', '[':']', '{':'}'}  

#     while True:

#         if len(stack) == 0 and len(temp) != 0:
#             return False

#         popped = stack.pop(-1)

#         if popped in '([{':
            
#             if len(temp) == 0 or temp[0] != dic[popped]:
#                 return False
                            
#             else:                
#                 temp = temp[1:]

#         else:
#             temp.insert(0,popped)

#         if len(stack) == 0 and len(temp)==0:
#             return True
        


# print(isValid(s))

'Notes: it works!'




'21. Merge Two Sorted Lists'

# Base
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


# Input

# 1st Input
# #List 1
# one1, two1, three1 = ListNode(1), ListNode(2), ListNode(4)
# one1.next, two1.next = two1, three1

# #List 2
# one2, two2, three2 = ListNode(1), ListNode(3), ListNode(4)
# one2.next, two2.next = two2, three2


# 2nd Input
# #List 1
# one1, two1, three1 = ListNode(4), ListNode(3), ListNode(4)
# one1.next, two1.next = two1, three1

# #List 2
# one2, two2, three2 = ListNode(1), ListNode(0), ListNode(50)
# one2.next, two2.next = two2, three2






#My Approach
# def mergeTwoLists(list1:ListNode, list2:ListNode) -> ListNode:

#     if list1.val == None and list2.val != None:
#         return list2
    
#     if list2.val == None and list1.val != None:
#         return list1
    
#     if list1.val == None and list2.val == None:
#         return ListNode(None)


#     head = ListNode()
#     curr_res = head

#     curr1, curr2 = list1, list2

#     while True:

#         if curr1 != None and curr2 != None:
            
#             if curr1.val <= curr2.val:
#                 curr_res.next = curr1
#                 curr_res = curr_res.next
#                 curr1 = curr1.next     
                
#             else:
#                 curr_res.next = curr2
#                 curr_res = curr_res.next
#                 curr2 = curr2.next                   

#         elif curr1 != None:
#             curr_res.next = curr1
#             curr_res = curr_res.next
#             curr1 = curr1.next

#         elif curr2 != None:
#             curr_res.next = curr2
#             curr_res = curr_res.next
#             curr2 = curr2.next
        

#         if curr1 == None and curr2 == None:
#             break


#     return head.next


# res = []
# res_node = mergeTwoLists(one1, one2)

# while res_node != None:

#     res.append(res_node.val)
#     res_node = res_node.next


# print(res)

'Notes: it works!'




'22. Generate Parentheses'

#Input
# n = 3   # Expected Output: ['((()))', '(()())', '(())()', '()(())', '()()()']

# # My Approach
# def generateParenthesis(n):
 
#     if n == 1:
#         return ['()']

#     result = []

#     for i in generateParenthesis(n-1):
#         result.append('('+ i +')')
#         result.append('()'+ i )
#         result.append(i + '()')


#     return sorted(set(result))

# print(generateParenthesis(4))

'''
Note: 
    My solution kind of work but it was missing one variation, apparently with DFS is solved.
'''


# # DFS Approach

# def generateParenthesis(n):
    
#     res = []

#     def dfs (left, right, s):

#         if len(s) == 2*n:
#             res.append(s)
#             return

#         if left < n:
#             dfs(left+1, right, s + '(')

#         if right < left:
#             dfs(left, right+1, s + ')')

#     dfs(0,0,'')

#     return res


# print(generateParenthesis(4))




'23. Merge k Sorted Lists'

# Base
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


# Input

# # 1st Input
# #List 1
# one1, two1, three1 = ListNode(1), ListNode(4), ListNode(5)
# one1.next, two1.next = two1, three1

# #List 2
# one2, two2, three2 = ListNode(1), ListNode(3), ListNode(4)
# one2.next, two2.next = two2, three2

# #List 3
# one3, two3 = ListNode(2), ListNode(6)
# one3.next = two3

# # List of lists
# li = [one1, one2, one3]

# My Approach

'''
Rationale:
  
    1. Create an empty node.
    2. Assign the node with the minimum value as next
    3. Move that node to its next node until reaches 'None'.
    4. When every value within the input list is None, breakout the loop and return.
'''

# def mergeKLists(lists:list[ListNode]) -> ListNode:
    
#     lists = [x for x in lists if x.val != '']

#     if len(lists) == 0:
#         return ListNode('')


#     head = ListNode('')
#     curr = head
#     li = lists

#     while True:

#         if li == [None]:
#             break

#         # Create a list of the current nodes in input that aren't None and sort them ascendingly by value
#         li = sorted([node for node in li if node != None], key = lambda x: x.val)

#         # Make the 'next_node' the next node to the curr None & move over to that node right away
#         curr.next = li[0]
#         curr = curr.next

#         # Move over to the next node of next_node
#         li[0] = li[0].next

#     return head.next


# res = mergeKLists([ListNode('')])
# res_li = []

# print(res)

'Notes: It worked'




'29. Divide Two Integers'

# Input

# Case 1
# dividend = 10
# divisor = 3

# Case 2
# dividend = 7
# divisor = -3

# My approach

'''
Rationale:
    1. Count how many times the divisor could be substracted from the dividend before reaching something smalle than the divisor
    2. if only one between dividend and the divisor is less than 0, the result would return a negative number 
'''


# def divide(dividend, divisor):
    
#     # case where 0 is divided by something
#     if dividend == 0:
#         return 0
    

#     # setting variables to operate
#     count = 0
#     div = abs(divisor)
#     dvnd = abs(dividend)


#     # case where the dividend is 1
#     if div == 1 and dvnd != 0:

#         if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0):
#             return -dvnd
        
#         else:
#             return dvnd
    

#     # case where the absolute divisor is greater than the dividend
#     if div > dvnd:
#         return 0
    
#     # case where both are the same number
#     if div == dvnd:
              
#         if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0):
#             return -1
        
#         else:
#             return 1
    
#     # case where is possible to divide iteratively
#     while dvnd >= div:

#         dvnd -= div
#         count += 1
    
#     # In case any is negative, the result will also be negative
#     if (dividend < 0 and divisor > 0) or (dividend > 0 and divisor < 0):
#         return -count

#     # Otherwise, just return
#     return count


# print(divide(dividend, divisor))

'Notes: My solution actually worked, theres nitpicking cases where it wont, but still '


# Another Approach

# def divide(dividend, divisor):

#     if (dividend == -2147483648 and divisor == -1): return 2147483647
            
#     a, b, res = abs(dividend), abs(divisor), 0

#     for x in range(32)[::-1]:

#         if (a >> x) - b >= 0:
#             res += 1 << x
#             a -= b << x
    
#     return res if (dividend > 0) == (divisor > 0) else -res

'Notes: This challenge is solved with bitwise operations '




'34. Find First and Last Position of Element in Sorted Array'

# Input

#case1
# nums = [5,7,7,8,8,10]
# target = 8  # Expected Output: [3,4]

#case2
# nums = [5,7,7,8,8,10]
# target = 6  # Expected Output: [-1,-1]

#case3
# nums = []
# target = 0  # Expected Output: [-1,-1]


# My Approach
# def searchRange(nums:list, target:int) -> int:
    
#     if target in nums:

#         starting_position = nums.index(target)

#         # The ending positions is calculated as of: (number of indices) - the relative position if the list is reversed
#         Ending_position = (len(nums)-1) - nums[::-1].index(target)

#         return [starting_position, Ending_position]

#     else:
#         return [-1,-1]

# print(searchRange(nums, target))

'Notes: It worked!'




'36. Valid Sudoku'

# Input

# Case 1
# board = [
# ["5","3",".",".","7",".",".",".","."]
# ,["6",".",".","1","9","5",".",".","."]
# ,[".","9","8",".",".",".",".","6","."]
# ,["8",".",".",".","6",".",".",".","3"]
# ,["4",".",".","8",".","3",".",".","1"]
# ,["7",".",".",".","2",".",".",".","6"]
# ,[".","6",".",".",".",".","2","8","."]
# ,[".",".",".","4","1","9",".",".","5"]
# ,[".",".",".",".","8",".",".","7","9"]
# ]

# Case 2
# board = [
# ["8","3",".",".","7",".",".",".","."]
# ,["6",".",".","1","9","5",".",".","."]
# ,[".","9","8",".",".",".",".","6","."]
# ,["8",".",".",".","6",".",".",".","3"]
# ,["4",".",".","8",".","3",".",".","1"]
# ,["7",".",".",".","2",".",".",".","6"]
# ,[".","6",".",".",".",".","2","8","."]
# ,[".",".",".","4","1","9",".",".","5"]
# ,[".",".",".",".","8",".",".","7","9"]
# ]


# My Approach

'''
Rationale:
    1. Pull out all the columns, rows and sub-boxes to be evaluated.
    2. Filter down empty colums, rows and sub-boxes.
    3. Cast set on each element on the 3 groups and 
        if one of them have less items than before the casting, return False. Otherwise, return True
'''

# def isValidSudoku(board: list[list[str]]) -> bool:

#     rows = board
#     columns = [list(x) for x in zip(*board)]


#     # Bulding the sub-boxes directly into the list
#         # Did it this way to save time complexity.
            
#     sub_boxes = [
#         [board[0][0:3],board[1][0:3],board[2][0:3]],
#         [board[0][3:6],board[1][3:6],board[2][3:6]],
#         [board[0][6:9],board[1][6:9],board[2][6:9]],
#         [board[3][0:3],board[4][0:3],board[5][0:3]],
#         [board[3][3:6],board[4][3:6],board[5][3:6]],
#         [board[3][6:9],board[4][6:9],board[5][6:9]],
#         [board[6][0:3],board[7][0:3],board[8][0:3]],
#         [board[6][3:6],board[7][3:6],board[8][3:6]],
#         [board[6][6:9],board[7][6:9],board[8][6:9]],
#     ]


#     # Validating rows
#     for row in rows:

#         row_wo_dot = [num for num in row if num != '.']

#         if len(row_wo_dot) != len(set(row_wo_dot)):
#             return False


#     # Validating columns
#     for col in columns:

#         col_wo_dot = [num for num in col if num != '.']

#         if len(col_wo_dot) != len(set(col_wo_dot)):
#             return False


#     # Validating Sub-boxes
#     for subb in sub_boxes:

#         plain_subb = [num for li in subb for num in li if num != '.']

#         if len(plain_subb) != len(set(plain_subb)):
#             return False


#     return True


# print(isValidSudoku(board))

'Notes: It works perfectly, but could be less verbose'




# Another Approach

# import collections

# def isValidSudoku(self, board):

#     rows = collections.defaultdict(set)
#     cols = collections.defaultdict(set)
#     subsquares = collections.defaultdict(set)

#     for r in range(9):

#         for c in range(9):

#             if(board[r][c] == "."):
#                 continue

#             if board[r][c] in rows[r] or board[r][c] in cols[c] or board[r][c] in subsquares[(r//3, c//3)]:
#                 return False
            
#             rows[r].add(board[r][c])
#             cols[c].add(board[r][c])
#             subsquares[(r//3,c//3)].add(board[r][c])

#     return True

'''
Notes: 
    This solution was much more elegant. And essentially the difference lays in this solution could be more scalable 
    since it builds the data holder while iterating.
'''




'38. Count and Say'

# Input

# Case 1
# n = 1   # Exp. Out: "1" (Base Case)

# Case 2
# n = 4   # Exp. Out: "1211" (Base Case)


# My approach

# 'Iterative suboptimal solution' 
# def countAndSay(n):

#     if n == 1:
#         return '1'
        
#     res = '1'

#     for _ in range(1, n):
    
#         pairs = []
#         count = 0
#         char = res[0]

#         for i in range(len(res)+1):

#             if i == len(res):
#                 pairs.append(str(count)+char)

#             elif res[i] == char:
#                 count += 1

#             else:       
#                 pairs.append(str(count)+char)
#                 char = res[i]
#                 count = 1

#         res = ''.join(pairs)
    
#     return res

# print(countAndSay(6))

'Notes: It works'




'Recursive Approach'
# def countAndSay(n):
#     if n == 1:
#         return '1'
#     return aux_countAndSay(countAndSay(n - 1))


# def aux_countAndSay(s):
   
#     if not s:
#         return ''
    
#     result = []
#     count = 1

#     for i in range(1, len(s)):

#         if s[i] == s[i - 1]:
#             count += 1

#         else:
#             result.append(str(count) + s[i - 1])
#             count = 1

#     result.append(str(count) + s[-1])

#     return ''.join(result)


# print(countAndSay(6))





'42. Trapping Rain Water'

# Input

# case 1
# height = [0,1,0,2,1,0,1,3,2,1,2,1]  # Exp. Out: 6

# case 2
# height = [4,2,0,3,2,5]  # Exp. Out: 9


# Solution

# def trap(height):

#     if not height:
#         return 0
    

#     left, right = 0, len(height)-1
#     left_max, right_max = 0, 0
#     result = 0

#     while left < right:

#         if height[left] < height[right]:

#             if height[left] >= left_max:
#                 left_max = height[left]

#             else:
#                 result += left_max - height[left]

#             left += 1
        
#         else:

#             if height[right] >= right_max:
#                 right_max = height[right]

#             else:
#                 result += right_max - height[right]

#             right -= 1
    
#     return result


# print(trap([3,0,2]))





'46. Permutations'

# Input

# Case 1
# nums = [1,2,3] # Exp. Out: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

# Case 2
# nums = [0,1] # Exp. Out: [[0,1],[1,0]]

# Case 3
# nums = [1] # Exp. Out: [[1]]


# Solution
# def permute(nums: list[int]) -> list[list[int]]:
    
#     if len(nums) == 0:
#         return []
    
#     if len(nums) == 1:
#         return [nums]
    
#     l = []

#     for i in range(len(nums)):

#         num = nums[i]
#         rest = nums[:i] + nums[i+1:]

#         for p in permute(rest):
#             l.append([num] + p)
        
#     return l





'48. Rotate Image'

# Input

# Case 1
# matrix = [
#     [1,2,3],
#     [4,5,6],
#     [7,8,9]
#     ]
# Exp. Out: [[7,4,1],[8,5,2],[9,6,3]]

# Case 2
# matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
# Exp. Out: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]


# # My approach
# def rotate(matrix: list[list[int]]):

#     n = len(matrix)

#     for i in range(n):

#         rot_row = []

#         for j in range(n):  # Since is given that is an squared matrix
#             rot_row.insert(0, matrix[j][i])

#         matrix.append(rot_row)
   
#     for i in range(n):
#         matrix.pop(0)

# rotate(matrix)
# print(matrix)

'Notes: It worked, but seems a little unorthodox'


# # Another Approach
# def rotate(matrix):

#     # reverse
#     # l = 0
#     # r = len(matrix) -1

#     # while l < r:
#     #     matrix[l], matrix[r] = matrix[r], matrix[l]
#     #     l += 1
#     #     r -= 1

#     x = ''
#     matrix = matrix[::-1]
#     x=0

#     # transpose 
#     for i in range(len(matrix)):
#         for j in range(i):
#             matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]


# rotate(matrix)
# print(matrix)

'Notes: This one looks more like a canon type solution'





'49. Group Anagrams'

# Input

# # Case 1
# strs = ["eat","tea","tan","ate","nat","bat"]
# #Exp. Out: [["bat"],["nat","tan"],["ate","eat","tea"]]

# # Case 2
# strs = [""]
# #Exp. Out: [[""]]

# # Case 3
# strs = ["a"]
# # Exp. Out: [["a"]]

# # Custom Case
# strs = ["ddddddddddg","dgggggggggg"]
# # Expected: [["dgggggggggg"],["ddddddddddg"]]



# My Approach

'''
Intuition:
    1. Take the first element of the input and make a list with all element that contains the same characters
    2. Erase the taken elements from the input.
    3. Reiterate steps 1 & 2 until the input is exhausted

'''

# def groupAnagrams(strs:list):
    
#     if len(strs) == 1:
#         return[strs]

#     # Auxiliary anagram checker
#     def is_anagram(ref:list, string:list):

#         if len(ref) != len(string):
#             return False

#         for char in ref:
            
#             if ref.count(char) != string.count(char):   
#                 return False

#         return True
    
#     # Creating Flag to manage repetitions
#     strs = [[word, False] for word in strs]


#     result = []

#     for word in strs:
             
#         if word[1] == False:

#             anagrams = []
#             anagrams.append(word[0])            
#             word[1] = True

#             for rest in strs:

#                 if rest[1] == False:

#                     if is_anagram(word[0], rest[0]):
#                         anagrams.append(rest[0])
#                         rest[1] = True
        
#             result.append(anagrams)

#     return result



# print(groupAnagrams(strs))

'''
Notes: 
    It passed 72/126 cases, the case below broke the code: 
        strs = ["ddddddddddg","dgggggggggg"] / Output: [["ddddddddddg","dgggggggggg"]], Expected: [["dgggggggggg"],["ddddddddddg"]]

    After the fixture, it works but beat no one in efficiency
'''


# Another Approach

# def groupAnagrams(strs):
    
#     freq = {}

#     for word in strs:

#         newWord = ''.join(sorted(word))

#         if newWord not in freq:
#             freq[newWord] = []
        
#         freq[newWord].append(word)

#     return list(freq.values())


# print(groupAnagrams(strs))

'''
Notes: Absolutely more elegant solution
'''




'50. Pow(x, n)'

# Input

# # Case 1
# x = 2.00000
# n = 10
# # Output: 1024.00000

# # Case 2
# x = 2.10000
# n = 3
# # Output: 9.26100

# # Case 3
# x = 2.00000
# n = -2
# # Output: 0.25000

# # Custom Case
# x = 0.00001
# n = 2147483647
# # Output: ...


# My Approach
# def myPow(x, n):

#     if x == 0:
#         return 0
    
#     if n == 0:
#         return 1

#     res = 1

#     for _ in range(abs(n)):
#         res *= x

#     if n > 0:
#         return f'{res:.5f}'
    
#     else:        
#         return f'{(1/res):.5f}'


# print(myPow(x, n))

'Notes: it works, but broke memory with the case: x = 0.00001, n=2147483647, it is 95% of the solution'


# # Another Approach
# def myPow(x: float, n: int) -> float:

#     b = n

#     if x == 0:
#         return 0
    
#     elif b == 0:
#         return 1
    
#     elif b < 0:
#         b = -b
#         x = 1 / x
    

#     a = 1

#     while b > 0:

#         if b % 2 == 0:
#             x = x * x
#             b = b // 2

#         else:
#             b = b - 1
#             a = a * x
            
#     return a

# print(myPow(x, n))

'''
Notes: 
    This solution takes advantage of the property x^(2n) = (x^2)^n, 
    saving a lot of time reducing in half the calculations each time the exponent is even.
'''




'53. Maximum Subarray'

# Input

# # Case 1
# nums = [-2,1,-3,4,-1,2,1,-5,4]
# # Output: 6 / [4,-1,2,1]

# # Case 2
# nums = [1]
# # Output: 1 / [1]

# # Case 3
# nums = [5,4,-1,7,8]
# # Output: 23 / [5,4,-1,7,8]



# My approach

'''
Intuition: Brute forcing
    1. Store the sum of the max array
    2. From the max len down to len = 1, evaluate with a sliding window each sub array.
    3. Return the max sum
'''


# def maxSubArray(nums):

#     max_sum = sum(nums)

#     # Handling the special case len = 1
#     if len(nums) == 1:
#         return max_sum
    
    
#     max_element = max(nums)
#     nums_len = len(nums)-1
#     idx = 0

#     while nums_len > 1:

#         if idx + nums_len > len(nums):
#             nums_len -= 1
#             idx = 0
#             continue
            
#         sub_array = nums[idx:idx+nums_len]
#         sub_array_sum = sum(sub_array)

#         if sub_array_sum > max_sum:
#             max_sum = sub_array_sum
            
#         idx += 1

#     # Handling the case where one element is greater than any subarray sum
#     if max_element > max_sum:
#         return max_element
    
#     return max_sum


# print(maxSubArray(nums))
        

'Notes: My solution worked 94,7% of the cases, the time limit was reached.'




# Kadane's Algorythm (for max subarray sum)

# def maxSubArray(nums):

#     max_end_here = max_so_far = nums[0]

#     for num in nums[1:]:

#         max_end_here = max(num, max_end_here + num)
#         max_so_far = max(max_so_far, max_end_here)
    
#     return max_so_far


# print(maxSubArray(nums))

'Notes: Apparently it was a classic problem'





'54. Spiral Matrix'

# Input

# # Case 1
# matrix = [[1,2,3],[4,5,6],[7,8,9]]
# # Output: [1,2,3,6,9,8,7,4,5]

# # Case 2
# matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12]]
# # Output: [1,2,3,4,8,12,11,10,9,5,6,7]

# # Custom Case  / m = 1
# matrix = [[2,1,3]]
# # Output: [2,1,3]

# # Custom Case / n = 1
# matrix = [[2],[1],[3],[5],[4]]
# # Output: [2,1,3,5,4]

# # Custom Case
# matrix = [[1,2],[3,4]]
# # Output: [1,2,4,3]

# # Custom Case
# matrix = [[2,3,4],[5,6,7],[8,9,10],[11,12,13],[14,15,16]]
# # Output: [2,3,4,7,10,13,16,15,14,11,8,5,6,9,12]




# My approach

'''
Intuition:
    1. Handle Special Cases: m = 1 / n = 1 

    2. Make a while loop that runs until the input has no elements:
        a. Take all the subelements from the first element and append them individually to the result.
        b. Make a for loop and take the last subelement of each element and append them individually to the result
        c. Take all the subelements from the last element and append them in a reverse order individually to the result.
        d. Make a for loop and take the first subelement of each element and append them in a reverse order individually to the result.

     3. Return the result
'''

# def spiralOrder(matrix):

#     if len(matrix) == 1:
#         return matrix[0]
    
#     if len(matrix[0]) == 1:
#         return [num for vec in matrix for num in vec]
    

#     result = []

#     while len(matrix) != 0:

#         first_element = matrix.pop(0)
#         result += first_element

#         if len(matrix) == 0:
#             break

#         second_element = []
#         for elem in matrix:
#             second_element.append(elem.pop(-1))
#         result += second_element

#         third_element = matrix.pop(-1)
#         result += reversed(third_element)
        
#         if len(matrix) > 0 and len(matrix[0]) == 0:
#             break

#         fourth_element = []
#         for elem in matrix:
#             fourth_element.append(elem.pop(0))
#         result += reversed(fourth_element)

#     return result

# print(spiralOrder(matrix))


'Notes: it works up to 76% of the cases, but from here seems more like patching something that could be better designed'


# Another Approach

# def spiralOrder(matrix):
        
#         result = []

#         while matrix:
#             result += matrix.pop(0) # 1

#             if matrix and matrix[0]: # 2 
#                 for line in matrix:
#                     result.append(line.pop())

#             if matrix: # 3
#                 result += matrix.pop()[::-1]

#             if matrix and matrix[0]: # 4

#                 for line in matrix[::-1]:
#                     result.append(line.pop(0))

#         return result


'Notes: Same logic, better executed'





'55. Jump Game'

# Input

# # Case 1
# nums = [2,3,1,1,4]
# # Output: True

# # Case 2
# nums = [3,2,1,0,4]
# # Output: False

# # Custom Case
# nums = [2,0,0]
# # Output: 


# My Approach

'''
Intuition: Brute Force

    I will check item by item to determine if the end of the list is reachable

'''


# def canJump(nums:list[int]) -> bool:

#     # Corner case: nums.lenght = 1 / nums[0] = 0
#     if len(nums) == 1 and nums[0] == 0:
#         return True
    
#     idx = 0

#     while True:

#         idx += nums[idx]

#         if idx >= len(nums)-1:
#             return True
        
#         if nums[idx] == 0 and idx < len(nums)-1:
#             return False
        

# print(canJump(nums))

'Notes: This solution suffice 91,2% of the case'


#Backtrack Approach

# def canJump(nums: list[int]) -> bool:

#     if len(nums)==1:
#         return True  

#     #Start at num[-2] since nums[-1] is given
#     backtrack_index = len(nums)-2 
#     #At nums[-2] we only need to jump 1 to get to nums[-1]
#     jump =1  

#     while backtrack_index>0:
#         #We can get to the nearest lily pad
#         if nums[backtrack_index]>=jump: 
#             #now we have a new nearest lily pad
#             jump=1 
#         else:
#             #Else the jump is one bigger than before
#             jump+=1 
#         backtrack_index-=1
    
#     #Now that we know the nearest jump to nums[0], we can finish
#     if jump <=nums[0]: 
#         return True
#     else:
#         return False 

'Notes: Right now I am not that interested in learning bactktracking, that will be for later'





'56. Merge Intervals'

#Input

# # Case 1
# intervals = [[1,3],[2,6],[8,10],[15,18]]
# # Output: [[1,6],[8,10],[15,18]]

# # Case 2
# intervals = [[1,4],[4,5]]
# # Output: [[1,5]]

# # Custom Case
# intervals = [[1,4],[0,0]]
# # Output: [...]


# My Approach

'''
Intuition:

    - Check the second item of the element and the first of the next, 
    if they coincide, merge.
        (Through a While Loop)
'''

# def merge(intervals:list[list[int]]) -> list[list[int]]:

#     #Handling the corner case
#     if len(intervals) == 1:
#         return intervals

#     intervals.sort(key=lambda x: x[0])

#     idx = 0

#     while idx < len(intervals)-1:

#         if intervals[idx][1] >= intervals[idx+1][0]:

#             merged_interval = [[min(intervals[idx][0], intervals[idx+1][0]), max(intervals[idx][1], intervals[idx+1][1])]]
#             intervals = intervals[:idx] + merged_interval + intervals[idx+2:]
#             idx = 0

#         else:
#             idx += 1

#     return intervals

# print(merge(intervals))

'Note: My solution works but is not efficient, since it has to go over the whole array again'



# Some other Approach

# def merge(intervals):
#     """
#     :type intervals: List[List[int]]
#     :rtype: List[List[int]]
#     """
#     intervals.sort()

#     merge_intervals = []
#     curr_interval = intervals[0]

#     for interval in intervals[1:]:

#         if curr_interval[1] < interval[0]:
#             merge_intervals.append(curr_interval)
#             curr_interval = interval

#         else:
#             curr_interval[1] = max(curr_interval[1], interval[1])

#     merge_intervals.append(curr_interval)

#     return merge_intervals


# print(merge(intervals))





'62. Unique Paths'

'''
*1st Dynamic programming problem

Notes:

This problem was pretty similar to the one on Turing's tests, althought here is requested
to find a bigger scale of thar problem. The classic 'How many ways would be to get from x to y',

if the problem were only set to m = 2, it'd be solved with fibonacci, but sadly that was not the case,
here, Dynamic Programming was needed.

The problem is graphically explained here: https://www.youtube.com/watch?v=IlEsdxuD4lY

But the actual answer I rather take it from the leetCode's solutions wall, since is more intuitive to me.

'''

# Input

# # Case 1:
# m, n = 3, 7
# # Output: 28

# # Case 2:
# m, n = 3, 2
# # Output: 3


# Solution

# def uniquePaths(m: int, n: int) -> int:

#     # Handling the corner case in which any dimention is 0
#     if n == 0 or m == 0:
#         return 0


#     # Here the grid is initialized
#     result = [[0]*n for _ in range(m)]

#     # The first column of the grid is set to 1, since there is only (1) way to get to each cell of that column
#     for row in range(m):
#         result[row][0] = 1

#     # The first row of the grid is set to 1, since there is only (1) way to get to each cell of that row
#     for col in range(n):
#         result[0][col] = 1


#     # Here all the grid is traversed summing up the cells to the left and up, since are the only ways to get to the current cell
#     # The range starts in 1 since all the first column and row are populated, so the traversing should start in [1,1]
#     for i in range(1, m):

#         for j in range(1, n):

#             result[i][j] = result[i-1][j] + result[i][j-1]
    

#     # The bottom right cell will store all the unique ways to get there
#     return result[-1][-1]


# print(uniquePaths(m, n))





'66. Plus One'

# Input

# # Case 1
# digits = [1,2,3]
# # Output: [1,2,4]

# # Case 2
# digits = [4,3,2,1]
# # Output: [4,3,2,2]

# # Case 3
# digits = [9]
# # Output: [1,0]

# # Custom Case
# digits = [9,9,9]
# # Output: [1,0,0,0]


# My approach

'''
Intuition:
    - The case is simple, the catch is to handle the case "[9,9,9]"
'''

# def plusOne(digits: list[int]) -> list[int]:

#     idx = -1

#     while abs(idx) <= len(digits):
        
#         if abs(idx) == len(digits) and digits[idx] == 9:

#             digits[idx] = 1
#             digits.append(0)
#             break

#         if digits[idx] != 9:

#             digits[idx] += 1
#             break

#         digits[idx] = 0
#         idx -= 1

#     return digits


# print(plusOne(digits=digits))

'''
Notes: 
While this code works, there was an even cleverer approach - To convert the digits into a int, add 1 and return as a list of ints
this way, is avoided the handling of cases
'''

# A different Approach

# def plusOne(digits: list[int]) -> list[int]:

#     number = int(''.join([str(x) for x in digits]))
#     number += 1
    
#     return [int(x) for x in str(number)]


# print(plusOne(digits=digits))





'69. Sqrt(x)'

# Input

# # Case 1
# x = 4
# # Output: 2

# # Case 2
# x = 8
# # Output: 2

# Custom Case
x = 399
# Output: ..




# My Approach

# limit = 46341

# # Auxiliary Eratosthenes' sieve function
# def primes(cap):  

#     primes = []
#     not_primes = []

#     for i in range(2, cap+1):

#         if i not in not_primes:
#             primes.append(i)
#             not_primes.extend([x for x in range(i*i, cap+1, i)])

#     return primes


# def mySqrt(x:int) -> int:

#     #Setting a limit for calculating primes
#     limit = x//2

#     prime_nums = primes(limit)

#     squares = list(map(lambda x: x*x, prime_nums))


#     #proning in the squares the correct range to make a range to evaluate
#     root_range = []
#     for idx, v in enumerate(squares):

#         if x <= v:
#             root_range = [prime_nums[idx-1], prime_nums[idx]]
#             break


#     #Calculating manually the square of each root in range to select the floor-root for the value
#     for root in range(root_range[0], root_range[1]+1):
        
#         if root*root >= x:
#             return result
        
#         result = root


# print(mySqrt(x))

'Notes: This approach was too complicated and actually no as efficient. Apparently with the notion of binary search is easier to solve'


# # Binary Search Approach

# def mySqrt(x):

#     left = 0
#     right = x

#     while left <= right:

#         mid = (left + right)//2

#         if mid*mid < x:
#             left = mid + 1

#         elif mid*mid > x: 
#             right = mid -1

#         else:
#             return mid
    
#     return right
    

# print(mySqrt(x))






'xxx'














