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





'xxx'

















