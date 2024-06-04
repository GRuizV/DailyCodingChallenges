'79. Word Search'

# Input

# # Case 1
# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = 'ABCCED'
# # Output: True


# # Case 2
# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = 'SEE'
# # Output: True


# # Case 3
# board = [["A","B","C","E"],["S","F","C","S"],["A","D","E","E"]]
# word = 'ABCB'
# # Output: False


'''
Intuition:

    The problem can be solved by traversing the grid and performing a depth-first search (DFS) for each possible starting position. 
    At each cell, we check if the current character matches the corresponding character of the word. 
    If it does, we explore all four directions (up, down, left, right) recursively until we find the complete word or exhaust all possibilities.

    Approach

        1. Implement a recursive function backtrack that takes the current position (i, j) in the grid and the current index k of the word.
        2. Base cases:
            - If k equals the length of the word, return True, indicating that the word has been found.
            - If the current position (i, j) is out of the grid boundaries or the character at (i, j) does not match the character at index k of the word, return False.
        3. Mark the current cell as visited by changing its value or marking it as empty.
        4. Recursively explore all four directions (up, down, left, right) by calling backtrack with updated positions (i+1, j), (i-1, j), (i, j+1), and (i, j-1).
        5. If any recursive call returns True, indicating that the word has been found, return True.
        6. If none of the recursive calls returns True, reset the current cell to its original value and return False.
        7. Iterate through all cells in the grid and call the backtrack function for each cell. If any call returns True, return True, indicating that the word exists in the grid. Otherwise, return False.
        
'''


# # Backtracking (Recursive) Approach
# def exist(board: list[list[str]], word: str) -> bool:


#     def backtrack(i, j, k):

#         if k == len(word):
#             return True
        
#         if i < 0 or i >= len(board) or j < 0 or j >= len(board[0]) or board[i][j] != word[k]:
#             return False
        
#         temp = board[i][j]
#         board[i][j] = ''
        
#         if backtrack(i+1, j, k+1) or backtrack(i-1, j, k+1) or backtrack(i, j+1, k+1) or backtrack(i, j-1, k+1):
#             return True
        
#         board[i][j] = temp

#         return False
        


#     for i in range(len(board)):

#         for j in range(len(board[0])):

#             if backtrack(i, j, 0):

#                 return True
            

#     return False
        

# print(exist(board, word))




'88. Merge Sorted Array'

# Input

# # Case 1
# nums1 = [1,2,3,0,0,0]
# m = 3
# nums2 = [2,5,6]
# n = 3
# # Output: [1,2,2,3,5,6]

# # Case 2
# nums1 = [1]
# m = 1
# nums2 = []
# n = 0
# # Output: [1]

# # Case 3
# nums1 = [0]
# m = 0
# nums2 = [1]
# n = 1
# # Output: [1]

# # Custom case
# nums1 = [0,2,0,0,0,0,0]
# m = 2
# nums2 = [-1,-1,2,5,6]
# n = 5
# # Output: [1]

# # Custom case
# nums1 = [-1,1,0,0,0,0,0,0]
# m = 2
# nums2 = [-1,0,1,1,2,3]
# n = 6
# # Output: [1]



# Solution

# def merge(nums1, m, nums2, n):

#     if m == 0:
#         for i in range(n):
#             nums1[i] = nums2[i]

#     elif n != 0:

#         m = n = 0

#         while n < len(nums2):

#             if nums2[n] < nums1[m]:

#                 nums1[:m], nums1[m+1:] = nums1[:m] + [nums2[n]], nums1[m:-1]

#                 n += 1
#                 m += 1
            
#             else:

#                 if all([x==0 for x in nums1[m:]]):
#                     nums1[m] = nums2[n]
#                     n += 1
                    
#                 m += 1


# merge(nums1,m,nums2,n)

# print(nums1)





'91. Decode Ways'  

# Input

# # Case 1:
# s = '12'
# # Output: 2

# # Case 2:
# s = '226'
# # Output: 3

# # Case 3:
# s = '06'
# # Output: 0

# # Custom Case:
# s = '112342126815'
# # Output: 11



# My apporach

# def fib(n):

#     res = [1,1]

#     for _ in range(n-1):
#         res.append(res[-2] + res[-1])
          
#     return res[1:]


# def numDecodings(s:str) -> int:

#     if s[0] == '0':
#         return 0
    
#     if len(s) == 1:
#         return 1

#     substrings = []
#     subs = ''

#     if s[0] in ['1', '2']:
#         subs += s[0]

#     for i in range(1, len(s)+1):

#         if i == len(s):
#             if subs != '':
#                 substrings.append(subs)

#         elif (s[i] in ['1', '2']) or (s[i-1] in ['1', '2'] and s[i] <= '6'):
#             subs += s[i]

#         else:
#             substrings.append(subs)
#             subs = ''

#     cap = len(max(substrings, key=len))
#     possibilities = fib(cap)

#     res = 0

#     for i in substrings:

#         if i in '10' or '20':
#             res += 1

#         else:
#             res += possibilities[len(i)-1] 
    
#     return res


# print(numDecodings(s))


'''
Notes: 
    This solution met 48% of expected results, there are a couple of cases I left unanalyzed.
    Nevertheless, the logic of fibonaccying the parsing numbers works, perhaps with more time
    a solution through this approach could work.

'''



# Dynamic Programming Approach

# def numDecodings(self, s):
    
#     dp = {len(s):1}

#     def backtrack(i):

#         if i in dp:
#             return dp[i]

#         if s[i]=='0':
#             return 0

#         if i==len(s)-1:
#             return 1

#         res = backtrack(i+1)

#         if int(s[i:i+2])<27:
#             res+=backtrack(i+2)
            
#         dp[i]=res

#         return res

#     return backtrack(0)





'98. Validate Binary Search Tree' 

# Base

# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right



# Input

# # Case 1
# root_layout = [2,1,3]
# root = TreeNode(val=2, left=TreeNode(val=1), right=TreeNode(val=3))
# # Output: True


# # Case 2
# root_layout  = [5,1,4,None, None, 3, 6]
# left = TreeNode(val=1)
# right = TreeNode(val=4, left=TreeNode(val=3), right=TreeNode(val=6)) 
# root = TreeNode(val=5, left=left, right=right)
# # Output: False


# # Custom Case 1
# root_layout  = [4,2,5,1,8,5,9,3,10,2,15]

# root = TreeNode(val=4)
# first_left, first_right = TreeNode(val=2), TreeNode(val=5)

# fl_left = TreeNode(val=1)
# fl_right = TreeNode(val=8, left=TreeNode(val=5), right=TreeNode(val=9)) 
# fr_left = TreeNode(val=3)
# fr_right = TreeNode(val=10, left=TreeNode(val=2), right=TreeNode(val=15)) 

# first_left.left, first_left.right = fl_left, fl_right
# first_right.left, first_right.right = fr_left, fr_right

# root.left, root.right = first_left, first_right
# # Output: True


# # Custom Case 2
# root_layout  = [10,9,11,3,4,7,15,8,4,13,16,12,21]

# root = TreeNode(val=10)
# first_left, first_right = TreeNode(val=9), TreeNode(val=11)

# fl_left = TreeNode(val=3, left=TreeNode(val=4), right=TreeNode(val=7))
# fl_right = TreeNode(val=15)
# fr_left = TreeNode(val=8, left=TreeNode(val=4), right=TreeNode(val=13))
# fr_right = TreeNode(val=16, left=TreeNode(val=12), right=TreeNode(val=21)) 

# first_left.left, first_left.right = fl_left, fl_right
# first_right.left, first_right.right = fr_left, fr_right

# root.left, root.right = first_left, first_right
# # Output: False


# # Custom Case 3
# root_layout  = [2,2,2]
# root = TreeNode(val=2, left=TreeNode(val=2), right=TreeNode(val=2))
# # Output: False


# My approach

'''
Intuition:
    traverse with DFS and check each (root-child) group,
    if balanced, check the next group, else, return False.

    if we get to the end of the tree and there were no imbalance, return True.

'''


# def dfs(root:TreeNode):

#     stack = [root]

#     while stack:

#         node = stack.pop()
#         ndv = node.val

#         if node.left or node.right:

#             if node.left:

#                 ndlv = node.left.val
                
#                 if node.left.val > node.val:
#                    return False
                
#                 stack.append(node.left)
            

#             if node.right:

#                 ndrv = node.right.val
                                                
#                 if node.right.val < node.val:
#                    return False
                
#                 stack.append(node.right)

#             if node.val == node.right.val and node.val == node.left.val:
#                 return False
            
#     return True


# print(dfs(root))

'Note: My solution works up to 78% of the cases'


# Inorder Tree Traversal Approach

# path = []

# def inorder(root:TreeNode, route:list):

#     if root is None:
#         return
    
#     inorder(root.left, route)
#     route.append(root.val)
#     inorder(root.right, route)


# inorder(root=root, route=path)


# print(path)

'Note: The Trick here is that the inorder traversal, basically returns a sorted list if is balanced!'





'101. Symmetric Tree' 

# Base

# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# Input

# # Case 1
# root_layout = [1,2,2,3,4,4,3]

# root = TreeNode(val=1)
# first_left= TreeNode(val=2, left=TreeNode(val=3), right=TreeNode(val=4))
# first_right = TreeNode(val=2, left=TreeNode(val=4), right=TreeNode(val=3))

# root.left, root.right = first_left, first_right
# # Output: True

# # Case 2
# root_layout = [1,2,2,None,3,None,3]

# root = TreeNode(val=1)
# first_left= TreeNode(val=2, right=TreeNode(val=3))
# first_right = TreeNode(val=2, right=TreeNode(val=3))

# root.left, root.right = first_left, first_right
# # Output: False

# # Custom Case 1
# root_layout = [1,2,2,2,None,2]

# root = TreeNode(val=1)
# first_left= TreeNode(val=2, left=TreeNode(val=2))
# first_right = TreeNode(val=2, left=TreeNode(val=2))

# root.left, root.right = first_left, first_right
# # Output: False



# My approach

'''
Intuition:
    Return a inorder-traversal list of the trees from the first left and right node,
    and one should be the reverse of the other.

    Handling corner cases:
    - If only a root: True
    - If only a root with two leaves, if the leaves are equal: True
    - If the number of nodes is even: False
'''

# def isSymetric(root:TreeNode):

#     tree_nodes = []

#     def inorder(root):

#         if root == None:
#             return 
        
#         inorder(root.left)
#         tree_nodes.append(root.val)
#         inorder(root.right)

#     inorder(root=root)

    
#     if len(tree_nodes) == 1:
#         return True
    
#     # If there are an even number of nodes, it can be symetrical
#     if len(tree_nodes)%2 == 0:
#         return False   
    
#     if len(tree_nodes) == 3:
#         if root.left.val == root.right.val:
#             return True

#     mid = len(tree_nodes)//2 
#     left_tree = tree_nodes[:mid]
#     right_tree = tree_nodes[mid+1:]
    
#     return left_tree == list(reversed(right_tree))


# print(isSymetric(root))

'Note: This solution works for cases where all node are identical, since it didnt distinguish between left and right'



# Recursive Approach

# def is_mirror(self, n1, n2):

#     if n1 is None and n2 is None:
#         return True
    
#     if (n1 is None) or (n2 is None) or (n1.val != n2.val):
#         return False

#     return self.is_mirror(n1.left, n2.right) and self.is_mirror(n1.right, n2.left)


# def isSymmetric(self, root):

#     return self.is_mirror(n1=root.left, n2=root.right)

# 'This solution works perfectly'





'102. Binary Tree Level Order Traversal' 

# Input

# Base

# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# Input

# # Case 1
# root_layout = [3,9,20,None,None,15,7]

# root = TreeNode(val=3)
# first_left= TreeNode(val=9)
# first_right = TreeNode(val=2, left=TreeNode(val=15), right=TreeNode(val=7))

# root.left, root.right = first_left, first_right
# # Output: [[3],[9,20],[15,7]]

# # Case 2
# root_layout = [1]
# root = TreeNode(val=1)
# # Output: [[1]]

# # Case 3
# root_layout = []
# # Output: []


# My Approach

'''
Intuition:

    With bread-first search, I can pull the values in order by levels.

    Given that Binary tree are binary, with the powers of 2
    it could be calculated how many nodes exist in each level.

    and with the l = 1 + floor(log_2(n)), the number of levels can
    be known just having the number of nodes.

    
'''
# from collections import deque
# from math import floor, log2

# def bfs(root:TreeNode):

#     queue = deque()
#     queue.append(root)

#     path = []

#     while queue:

#         node = queue.popleft()

#         if node not in path:

#             path.append(node)

#             if node.left:
#                 queue.append(node.left)

#             if node.right:
#                 queue.append(node.right)

#     return [x.val for x in path]

# nodes_list = bfs(root=root)

# n_levels = 1 + floor(log2(len(nodes_list)))

# result = []

# for i in range(n_levels):

#     temp = []

#     for j in range(pow(2, i)):

#         if nodes_list:
#             temp.append(nodes_list.pop(0))
    
#     result.append(temp)
    

# print(result)


'Notes: This solution works but the leetcode interpreter didnt recognized the log2 function'


# A Simplier Approach

# def levelsOrder(root:TreeNode):

#     from collections import deque
    
#     queue = deque()
#     queue.append(root)    
#     result = []

#     while queue:

#         queue_len = len(queue)
#         level = [] 
        
#         for i in range(queue_len):

#             node = queue.popleft()

#             if node is not None:

#                 level.append(node.val)
#                 queue.append(node.left)
#                 queue.append(node.right)

#         if level:   
#             result.append(level)

#     return result

# print(levelsOrder(root=root))

'Done'





'103. Binary Tree Zigzag Level Order Traversal' 

# Base

# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# Input

# # Case 1
# root_layout = [3,9,20,None,None,15,7]

# root = TreeNode(val=3)
# first_left= TreeNode(val=9)
# first_right = TreeNode(val=20, left=TreeNode(val=15), right=TreeNode(val=7))

# root.left, root.right = first_left, first_right
# # Output: [[3],[20,9],[15,7]]

# # Case 2
# root_layout = [1]
# root = TreeNode(val=1)
# # Output: [[1]]

# # Case 3
# root_layout = []
# # Output: []


# My Approach

'''
Notes:
    This will go apparently the same as the level order, but in the other way arround
    and this time is alternating depending of the level
'''


# def zigzagLevelOrder(root:TreeNode) -> list[list[int]]:

#     from collections import deque

#     queue = deque()
#     queue.append(root)
#     result = []
#     level = 1

#     while queue:

#         len_q = len(queue)
#         level_nodes = []
      
#         for i in range(len_q):

#             node = queue.popleft()

#             if node is not None:

#                 queue.append(node.left)
#                 queue.append(node.right)
#                 level_nodes.append(node.val)

#         if len(level_nodes) != 0:

#             if level % 2 == 0:
#                 level_nodes = list(reversed(level_nodes))
            
#             result.append(level_nodes)
        
#         level += 1
    
#     return result

# print(zigzagLevelOrder(root=root))

'It worked!'





'104. Maximum Depth of Binary Tree'

# Base

# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# Input

# # Case 1
# root_layout = [3,9,20,None,None,15,7]

# root = TreeNode(val=3)
# first_left= TreeNode(val=9)
# first_right = TreeNode(val=20, left=TreeNode(val=15), right=TreeNode(val=7))

# root.left, root.right = first_left, first_right
# # Output: 3

# # Case 2
# root_layout = [1, None, 2]

# root = TreeNode(val=1, right=TreeNode(val=2))
# # Output: 2


# My approach

'''
Notes:
    Here could be to ways (or more) to solve it:
        1. Implement the BFS by level listing (like the challenges prior to this one) and count the elements of the result
        2. Simply list through DFS or BFS and apply l = 1 + floor(log_2(n)), to know the number of levels, but probably leetcode won't have 
           the log2 function in its math module, so I'll the first way.
'''

# def maxDepth(root:TreeNode) -> int:

#     from collections import deque

#     queue = deque()
#     queue.append(root)
#     result = []

#     while queue:

#         queue_len = len(queue)
#         level = []

#         for _ in range(queue_len):

#             node = queue.popleft()

#             if node is not None:

#                 queue.append(node.left)
#                 queue.append(node.right)

#                 level.append(node.val)

#         if level:
#             result.append(level)
    
#     return result

# print(maxDepth(root))

'Done!'





'105. Construct Binary Tree from Preorder and Inorder Traversal'

# # Base

# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# # Input

# # Case 1
# preorder, inorder = [3,9,20,15,7],[9,3,15,20,7]
# # Output: [3,9,20,None,None,15,7]


# def buildTree(preorder, inorder):

#     if inorder:

#         idx = inorder.index(preorder.pop(0))
#         root = TreeNode(val = inorder[idx])
#         root.left = buildTree(preorder=preorder, inorder=inorder[:idx])
#         root.right = buildTree(preorder=preorder, inorder=inorder[idx+1:])

#         return root


'Done'





'108. Convert Sorted Array to Binary Search Tree'

# # Base

# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right


# Input

# # Case 1
# nums = [-10,-3,0,5,9]
# # Output: [0,-3,9,-10,None,5] | [0,-10,5,None,-3,None,9]

# # Case 2
# nums = [1,3]
# # Output: [3,1] | [1,None,-3]


# My Approach

'''
Intuition:
    Learnt for the prior exercise, the middle node will be taken as the root.
    from there, it can recursively built the solution.
        base case = when len(nums) = 0
'''

# def sortedArrayToBST(nums:list[int]) -> TreeNode:

#     nums_len = len(nums)

#     if nums_len:

#         idx = nums_len // 2

#         return TreeNode(val = nums[idx], left = sortedArrayToBST(nums=nums[:idx]), right = sortedArrayToBST(nums=nums[idx+1:]))



# node = sortedArrayToBST(nums=nums)
# print(node)

'Done'





'116. Populating Next Right Pointers in Each Node'

# Base

# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None, next=None):
#         self.val = val
#         self.left = left
#         self.right = right
#         self.next = next


# Input

# # Case 1
# tree_lauout = [1,2,3,4,5,6,7]

# left = TreeNode(val=2, left=TreeNode(val=4), right=TreeNode(val=5))
# right = TreeNode(val=3, left=TreeNode(val=6), right=TreeNode(val=7))
# root = TreeNode(val=1, left=left, right=right)
# # Output: [1,#,2,3,#,4,5,6,7,#]


#My Approach

'''
Intuition:
    This could be solved with the BFS modified to catch nodes by level,
    and with the level picked from each loop, modify its pointers in that order 
'''

# def connect(root:TreeNode) -> TreeNode:
    
#     #Start
#     queue = [root]
    
#     while queue:

#         q_len = len(queue)
#         level = []

#         for i in range(q_len):

#             node = queue.pop(0)

#             if node:

#                 queue.extend([node.left, node.right])
#                 level.append(node)
        
#         if level:

#             for i in range(len(level)):

#                 if i != len(level)-1:

#                     level[i].next = level[i+1]
    
#     return root

'Worked right way, YAY! :D'






'''118. Pascal's Triangle'''

#Input

# #case 1
# numRows = 5
# #Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]

# #case 2
# numRows = 1
# #Output: [[1]]

# My Approach

'''
Intuition:
    initialize a preset solution to [[1],[1,1]] and according to the
    parameter passed in the function, start to sum and populate this sums to a list
    like [1]+[resulting_sums]+[1] and return that back to the preset solution, to operate over that
    new element,

        The number of loops will be numRows - 2 (given the 2 initial elements)
'''

# def generate(numRows:int) -> list[list[int]]:

#     result = [[1],[1,1]]

#     if numRows == 1:
#         return [result[0]]
    
#     if numRows == 2:
#         return result
    

#     for i in range(1, numRows-1):

#         new_element = []

#         for j in range(i):
#             new_element.append(result[-1][j]+result[-1][j+1])

#         if new_element:
#             result.append([1]+new_element+[1])

#     return result

# print(generate(numRows=5))

'It worked!'






'''121. Best Time to Buy and Sell Stock'''

# Input

# #Case 1
# prices = [7,1,5,3,6,4]
# #Output: 5

# #Case 2
# prices = [7,6,4,3,1]
# #Output: 0



# My approach

'''
Intuition
    - Corner Case: if is a ascendingly sorted list, return 0.
    
    - Pick the first item and set the profit as the max between the current profit and the difference between the first element
      the max value from that item forward.
    
    Do this in a while loop until len(prices) = 1.
'''


# def maxProfit(prices: list[int]) -> int:

#     profit = 0

#     if prices == sorted(prices, reverse=True):
#         return profit
    

#     while len(prices) > 1:

#         purchase = prices.pop(0)

#         profit = max(profit, max(prices)-purchase)
    
#     return profit


# print(maxProfit(prices=prices))

'This approach met 94% of the results'


# Kadane's Algorithm

# def maxProfit(prices: list[int]) -> int:

#     buy = prices[0]
#     profit = 0

#     for num in prices[1:]:

#         if num < buy:
#             buy = num
        
#         elif num-buy > profit:
#             profit = num - buy
    
    
#     return profit  



# print(maxProfit(prices=prices))

'Done'






'''122. Best Time to Buy and Sell Stock II'''

#Input

# #Case 1
# prices = [7,1,5,3,6,4]
# #Output: 7

# #Case 2
# prices = [1,2,3,4,5]
# #Output: 4

# #Case 3
# prices = [7,6,4,3,1]
# #Output: 0

# #Custom Case
# prices = [3,3,5,0,0,3,1,4]
# #Output: 0

# # My approach
# def maxProfit(prices:list[int]) -> int:

#     if prices == sorted(prices, reverse=True):
#         return 0
    
#     buy = prices[0]
#     buy2 = None
#     profit1 = 0
#     profit2 = 0
#     total_profit = 0

#     for i in range(1, len(prices)):

#         if prices[i] < buy:
#             buy = prices[i]
        
#         elif prices[i] - buy >= profit1:            
#             profit1 = prices[i] - buy
#             buy2 = prices[i] 

#             for j in range(i+1, len(prices)):

#                 if prices[j] < buy2:
#                     buy2 = prices[j]

#                 elif prices[j] - buy2 >= profit2:
#                     profit2 = prices[j] - buy2
#                     total_profit = max(total_profit, profit1 + profit2)
        
#         total_profit = max(total_profit, profit1)

#     return total_profit


# print(maxProfit(prices=prices))

'This solution went up to solve 83% of the cases, the gap was due to my lack of understanding of the problem'


# # Same Kadane's but modified
# def maxProfit(prices:list[int]) -> int:

#     max = 0 
#     start = prices[0]
#     len1 = len(prices)

#     for i in range(0 , len1):

#         if start < prices[i]: 
#             max += prices[i] - start

#         start = prices[i]

#     return max


# print(maxProfit(prices=prices))

'My mistake was to assume it can only be 2 purchases in the term, when it could be as many as it made sense'






'''124. Binary Tree Maximum Path Sum'''

# # Base 
# class TreeNode(object):
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

#Input

# #Case 1
# tree_layout = [1,2,3]
# root = TreeNode(val=1, left=TreeNode(val=2), right=TreeNode(val=3))
# #Output: 6

# #Case 2
# tree_layout = [-10,9,20,None, None,15,7]
# left = TreeNode(val=9)
# right = TreeNode(val=20, left=TreeNode(val=15), right=TreeNode(val=7))
# root = TreeNode(val=-10, left=left, right=right)
# #Output: 42

# #Custom Case
# tree_layout = [1,-2,3,1,-1,-2,-3]
# left = TreeNode(val=-2, left=TreeNode(val=1), right=TreeNode(val=3))
# right = TreeNode(val=-3, left=TreeNode(val=-2, left=TreeNode(val=-1)))
# root = TreeNode(val=1, left=left, right=right)
# #Output: 3


#My approach

'''
Intuition:
    - Make a preorder traversal tree list.
    - Apply Kadane's algorithm to that list.
'''


# def maxPathSum(root:TreeNode) -> int:

#     #First, Preorder
#     path = []

#     def preorder(node:TreeNode) -> None:

#         if node:
#             preorder(node=node.left)
#             path.append(node.val)
#             preorder(node=node.right)

#     preorder(node=root)

#     #Now Kadane's
#     max_so_far = max_end_here = path[0]

#     for num in path[1:]:

#         max_end_here = max(num, max_end_here + num)
#         max_so_far = max(max_so_far, max_end_here)

#     return max_so_far


# print(maxPathSum(root=root))

'''
Notes:
    - On the first run it went up to 59% of the cases, thats Kudos for me! :D
    - The problem with this algorithm is that it supposes that after reaching a parent and child node,
      it's possible to go from a right child to the parent of the parent and that either forcibly makes
      to pass twice from the parent before going to the granparent, or that one grandchild is connected
      to the grandfather, which is also out of the rules.

      I misinterpret this because one of the examples showed a path [leftchild, parent, rightchild] which
      is valid only if we don't want to pass thruough the grandparent.
    
    The best choice here is to make a recursive proning algorithm
'''


# #A recursive approach
# def maxPathSum(root):

#     max_path = float('-inf') #Placeholder

#     def get_max_gain(node):

#         nonlocal max_path

#         if not node:
#             return 0
        
#         gain_on_left = max(get_max_gain(node.left),0)
#         gain_on_right = max(get_max_gain(node.right),0)

#         current_max_path = node.val + gain_on_left + gain_on_right
#         max_path = max(max_path, current_max_path)

#         return node.val + max(gain_on_left, gain_on_right)
    
#     get_max_gain(root)

#     return max_path

# print(maxPathSum(root))
'Done'






'''125. Valid Palindrome'''

# def isPalindrome(s:str) -> bool:

#     s = ''.join([x for x in s if x.isalpha()]).casefold()

#     return s == s[::-1]



# a = '0P'

# a = ''.join([x for x in a if x.isalnum()]).casefold()

# print(a)
'Done'






'''127. Word Ladder'''

#Input

# #Case 1
# begin_word, end_word, word_list = 'hit', 'cog', ['hot', 'dot', 'dog', 'lot', 'log', 'cog']
# #Output: 5

# #Custom Case
# begin_word, end_word, word_list = 'a', 'c', ['a', 'b', 'c']
# #Output: 5


# My approach

'''
Intuition:
    1. handle the corner case: the end_word not in the word_list
    2. create an auxiliary func that check the word against the end_word: True if differ at most by 1 char, else False.
    3. create a counter initialized in 0
    4. start checking the begin_word and the end_word, if False sum 1 to the count, and change to the subquent word in the word_list and do the same.
'''

# def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:

#     if endWord not in wordList:
#         return 0
    
#     def check(word):
#         return False if len([x for x in word if x not in endWord]) > 1 else True
       
#     if beginWord not in wordList:
#         wordList.insert(0,beginWord)
#         count = 0
    
#     else:
#         count = 1
    
#     for elem in wordList:
#         count += 1

#         if check(elem):
#             return count     
            
#     return 0


# print(ladderLength(beginWord=begin_word, endWord=end_word, wordList=word_list))


'This solution only went up to the 21% of the cases'


# bfs approach

# from collections import defaultdict, deque

# def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:

#     if endWord not in wordList or not endWord or not beginWord or not wordList:
#         return 0

#     L = len(beginWord)
#     all_combo_dict = defaultdict(list)

#     for word in wordList:
#         for i in range(L):
#             all_combo_dict[word[:i] + "*" + word[i+1:]].append(word) 

#     queue = deque([(beginWord, 1)])
#     visited = set()
#     visited.add(beginWord)

#     while queue:
#         current_word, level = queue.popleft()

#         for i in range(L):
#             intermediate_word = current_word[:i] + "*" + current_word[i+1:]

#             for word in all_combo_dict[intermediate_word]:

#                 if word == endWord:
#                     return level + 1

#                 if word not in visited:
#                     visited.add(word)
#                     queue.append((word, level + 1))
                    
#     return 0






'''128. Longest Consecutive Sequence'''


#Input

# #Case 1
# nums = [100,4,200,1,3,2]
# #Output: 4

# #Case 2
# nums = [0,3,7,2,5,8,4,6,0,1]
# #Output: 9


#My approach

# def longestConsecutive(nums:list)->int:

#     if not nums:
#         return 0
   
#     nums.sort()

#     sequences = {}

#     for i in range(len(nums)):

#         curr_seqs = [x for elem in sequences.values() for x in elem]

#         if nums[i] not in curr_seqs:

#             sequences[nums[i]] = [nums[i]]

#             for j in range(i+1,len(nums)):
                
#                 criteria = range( min(sequences[nums[i]])-1, max(sequences[nums[i]])+2)
#                 if nums[j] in criteria:
#                     sequences[nums[i]].append(nums[j])

#     result = max(sequences.values(), key=len)

#     return len(set(result))

# print(longestConsecutive(nums=nums))

'This solution went up to 83% of the cases'


# Another Approach

# def longestConsecutive (nums):

#     if not nums:
#         return 0
    
#     num_set = set(nums)

#     longest = 1

#     for num in nums:

#         count = 1

#         if num-1 not in num_set:

#             x = num

#             while x+1 in num_set:
               
#                 count+=1
#                 x+=1

#         longest = max(longest, count)

#     return longest

# print(longestConsecutive(nums=nums))






'''130. Surrounded Regions'''

#Input

# #Case 1
# board = [
#     ["X","X","X","X"],
#     ["X","O","O","X"],
#     ["X","X","O","X"],
#     ["X","O","X","X"]
#     ]
# # output = [
# #     ["X","X","X","X"],
# #     ["X","X","X","X"],
# #     ["X","X","X","X"],
# #     ["X","O","X","X"]
# #     ]

# #Case 2
# board = [
#     ['X']
#     ]
# # output = [
#     # ['X']
#     # ]

# #Custom Case
# board = [["O","O"],["O","O"]]



#My approach

'''
Intuition:
    1. Check if there is any 'O' at the boarders.
    2. Check is there is any 'O' adjacent to the one in the boarder:
        - If do, add them to the not-be-flipped ground and re run.
        - if doesn't, flip everything to 'X' and return
    (Do this until there is no 'O' unchecked )
'''

# def solve(board:list[list[str]]) -> None:

#     M = len(board)
#     N = len(board[0])

#     no_flip = []
#     all_os = []


#     # Collect all 'O's
#     for i in range(M):
#         all_os.extend((i,j) for j in range(N) if board[i][j] == 'O')
    

#     #   Check if there is a boarder 'O' within the group
#     for i in range(len(all_os)):

#         if all_os[i][0] in (0, M-1) or all_os[i][1] in (0, N-1):
#             no_flip.append(all_os[i])


#     # Collect the 'O's near to no_flip 'O' iteratively
#     flipped = None
#     i = 0

#     while True:

#         # Condition to end the loop
#         if len(all_os) == 0 or i == len(all_os) and flipped is False:
#             break

#         #Collecting the possibilities of an adjacent 'O'
#         adjacents = []

#         for pos in no_flip:
#             adjacents.extend([(pos[0]-1, pos[1]), (pos[0]+1, pos[1]), (pos[0], pos[1]-1), (pos[0], pos[1]+1)])
        
#         #Check if the current element is adjacent to any no_flip 'O'
#         if all_os[i] in adjacents:
#             no_flip.append(all_os.pop(i))
#             flipped = True
#             i = 0
#             continue

#         i += 1
#         flipped = False


#     # Rewritting the board
#     #   Resetting the board to all "X"
#     for i in range(M):
#         board[i] = ["X"]*N
    
#     #   preserving the no_flip 'O's
#     for o in no_flip:
#         board[o[0]][o[1]] = 'O'


# solve(board=board)

'This solution met 98.2% of the cases'


#DFS Approach

# def solve(board):

#     n,m=len(board),len(board[0])
#     seen=set()

#     def is_valid(i,j):
#         return 0 <= i < n and 0<= j <m and board[i][j]=="O" and (i,j) not in seen
    
#     def is_border(i,j):
#         return i == 0 or i == n-1 or j == 0 or j == m-1
    
#     def dfs(i,j):

#         board[i][j]="y"
#         seen.add((i,j))

#         for dx , dy in ((0,1) ,(0,-1) ,(1,0),(-1,0)):
#             new_i , new_j = dx + i , dy + j

#             if is_valid(new_i , new_j):
#                 dfs(new_i , new_j)
        
#     for i in range(n):
#         for j in range(m):
#             if is_border(i,j) and board[i][j]=="O":
#                 dfs(i,j) 
                
#     for i in range(n):
#         for j in range(m):
#             if board[i][j]=="y":
#                 board[i][j]="O"
#             else:
#                 board[i][j]="X"

# solve(board)






'''131. Palindrome Partitioning'''

#Input

# # Case 1
# s = 'aab'
# # Output: [["a","a","b"],["aa","b"]]

# # Custom Case
# s = 'aabcdededcbada'
# # Output: [["a","a","b"],["aa","b"]]



# My approach

'''
Intuition:

    Here I don't actually have much ideas in how to solve it, but one good approach
    I think woul dbe to make a function that can pull all the palindroms present in a string.

    that could be a good start point.
'''

# # Custom Case
# s = 'aabcdededcbada'
# # Output: ['abcdededcba', 'bcdededcb', 'cdededc', 'deded', 'ded', 'ede', 'ded', 'ada', 'aa'] 

# def palindromes(string:str) -> list[str]:

#     s_len = len(string)
#     palindromes = []

#     for i in range(s_len, 1, -1):   # from s_len down to length 2 of substring
       
#         j = 0

#         while j + i <= s_len: 

#             subs = string[j:j+i]

#             if subs == subs[::-1]:

#                 palindromes.append(subs)

#             j += 1

#     print(palindromes)



# # Printout: ['abcdededcba', 'bcdededcb', 'cdededc', 'deded', 'ded', 'ede', 'ded', 'ada', 'aa'] 
# palindromes(string=s)

'''
At least this I was able to do, from here on, I am feeling I am going to brute forcing this and it won't end up being efficient.

I didn't actually solved it but I don't want to waste more time over this
'''






'''134. Gas Station'''

# Input

# #Case 1
# gas, cost = [1,2,3,4,5], [3,4,5,1,2]
# #Output = 3

# #Case 2
# gas, cost = [2,3,4], [3,4,3]
# #Output = -1

# # #Custom Case 
# gas, cost = [3,1,1], [1,2,2]
# #Output = 0


# My Approach

'''
Intuition:
    - Handle the corner case where sum(gas) < sum(cos) / return -1
    - Collect the possible starting point (Points where gas[i] >= cost[i])
    - Iterate to each starting point (holding it in a placeholder) to check 
        if a route starting on that point completes the lap:
        
        - if it does: return that starting point
        - if it doesn't: jump to the next starting point

    - If no lap is completed after the loop, return -1.

'''

# def canCompleteCircuit(gas:list[int], cost:list[int]) -> int:

    
#     # Handle the corner case
#     if sum(gas) < sum(cost):
#         return -1
    

#     # Collect the potential starting stations
#     stations = [i for i in range(len(gas)) if gas[i] >= cost[i]]


#     # Checking routes starting from each collected station
#     for i in stations:

#         station = i
#         tank = gas[i]

#         while tank >= 0:
            
#             # Travel to the next station
#             tank = tank - cost[station] 

#             # Check if we actually can get to the next station with current gas
#             if tank < 0:
#                 break
                
#             # If we are at the end of the stations (clockwise)
#             if station + 1 == len(gas):
#                 station = 0
                        
#             else:
#                 station += 1
                        
#             #If we success in making the lap
#             if station == i:
#                 return i
        
#             # Refill the tank
#             tank = tank + gas[station]


#     # in case no successful loop happens, return -1
#     return -1

# print(canCompleteCircuit(gas=gas, cost=cost))

'My solution met 85% of the test cases'


# # Another approach

# def canCompleteCircuit(gas:list[int], cost:list[int]) -> int:

    
#     # Handle the corner case
#     if sum(gas) < sum(cost):
#         return -1
    
#     current_gas = 0
#     starting_index = 0

#     for i in range(len(gas)):

#         current_gas += gas[i] - cost[i]

#         if current_gas < 0:
#             current_gas = 0
#             starting_index = i + 1
            
#     return starting_index

# print(canCompleteCircuit(gas=gas, cost=cost))

'This simplified version prooved to be more efficient'





'''138. Copy List with Random Pointer'''

# # Base
# class Node:
#     def __init__(self, x, next=None, random=None):
#         self.val = int(x)
#         self.next = next
#         self.random = random


# #Input

# #Case 1
# head_map = [[7,None],[13,0],[11,4],[10,2],[1,0]]

# #Build the relations of the list
# nodes = [Node(x=val[0]) for val in head_map]

# for i in range(len(nodes)):
#     nodes[i].next = None if i == len(nodes)-1 else nodes[i+1]
#     nodes[i].random = None if head_map[i][1] is None else nodes[head_map[i][1]]

# head = nodes[0]

#Output: [[7,None],[13,0],[11,4],[10,2],[1,0]]


# #Case 2
# head_map = [[1,1],[2,1]]

# #Build the relations of the list
# nodes = [Node(x=val[0]) for val in head_map]

# for i in range(len(nodes)):
#     nodes[i].next = None if i == len(nodes)-1 else nodes[i+1]
#     nodes[i].random = None if head_map[i][1] is None else nodes[head_map[i][1]]

# head = nodes[0]

# #Output: [[7,None],[13,0],[11,4],[10,2],[1,0]]


# #Case 3
# head_map = [[3,None],[3,0],[3,None]]

# #Build the relations of the list
# nodes = [Node(x=val[0]) for val in head_map]

# for i in range(len(nodes)):
#     nodes[i].next = None if i == len(nodes)-1 else nodes[i+1]
#     nodes[i].random = None if head_map[i][1] is None else nodes[head_map[i][1]]

# head = nodes[0]

# #Output: [[7,None],[13,0],[11,4],[10,2],[1,0]]



# My Approach

'''
Intuition:
    - Traverse through the list
    - Create a copy of each node and store it into a list along side with the content of the random pointer.
    - Traverse the list linking each node to the next and the random pointer to the position in that list.

Thoughts:

- It is possible to create the list with a recursive solution but it'll be still necesary to traverse again
    to collect the content of the random pointer or how else I can point to somewhere at each moment I don't know if it exist. 

'''

# def copyRandomList(head:Node) -> Node:

#     # Handle the corner case where there is a single node list
#     if head.next == None:
#         result = Node(x = head.val, random=result)
#         return result

#     # Initilize a nodes holder dict to collect the new nodes while traversing the list
#     nodes = {}

#     # Initilize a nodes holder list to collect the old nodes values while traversing the list
#     old_nodes_vals = []

#     # Initialize a dummy node to traverse the list
#     current_node = head

#     # Traverse the list
#     while current_node is not None:

#         # Collect the old nodes
#         old_nodes_vals.append(current_node.val)

#         # Check if the node doesn't already exist due to the random pointer handling
#         if current_node.val not in nodes.keys(): 

#             new_node = Node(x = current_node.val)
#             nodes[new_node.val] = new_node
        
#         else:
#             new_node = nodes[current_node.val]


#         # Handle the random pointer 
#         if current_node.random is None:
#             new_node.random = None

#         else:

#             # If the randoms does not exist already in the dict, create a new entry in the dict with the random value as key and a node holding that value 
#             if current_node.random.val not in nodes.keys():
#                 nodes[current_node.random.val] = Node(x = current_node.random.val)
          
#             new_node.random = nodes[current_node.random.val]


#         # Move to the next node
#         current_node = current_node.next
    

#     # Pull the nodes as a list to link to their next attribute
#     nodes_list = [nodes[x] for x in old_nodes_vals]

#     # Traverse the nodes list
#     for i, node in enumerate(nodes_list):

#         node.next = nodes_list[i+1] if i != len(nodes_list)-1 else None
   

#     return nodes_list[0]


# result = copyRandomList(head=head)


# new_copy = []
# while result is not None:
#     new_copy.append([result.val, result.random.val if result.random is not None else None])
#     result = result.next


'My solution works while the values of the list are unique, otherwise a new approach is needed'


# Another Approach

# def copyRandomList(head:Node):

#     nodes_map = {}

#     current = head

#     while current is not None:

#         nodes_map[current] = Node(x = current.val)
#         current = current.next

    
#     current = head

#     while current is not None:

#         new_node = nodes_map[current]
#         new_node.next = nodes_map.get(current.next)
#         new_node.random = nodes_map.get(current.random)

#         current = current.next
    
#     return nodes_map[head]


# result = copyRandomList(head=head)


# new_copy = []
# while result is not None:
#     new_copy.append([result.val, result.random.val if result.random is not None else None])
#     result = result.next





'''139. Word Break'''

#Input

# #Case 1
# s = "leetcode" 
# wordDict = ["leet","code"]
# #Output: True

# #Case 2
# s = "applepenapple"
# wordDict = ["apple","pen"]
# #Output: True

# #Case 3
# s = "catsandog"
# wordDict = ["cats","dog","sand","and","cat"]
# #Output: False


# My Approach

'''
Intuition:
    (Brute-force): in a while loop go word for word in the dict checking if the 
        word exists in the string:

            - If it does: Overwrite the string taking out the found word / else: go to the next word

        The loop will be when either no words are found in the string or the string is empty

        if after the loop the string is empty, return True, otherwise False
'''

# def workBreak(string:str, word_dict:list[str]) -> bool:

#     j = 0
#     while j < len(word_dict):

#         if word_dict[j] in string:

#             w_len = len(word_dict[j])
#             idx = string.find(word_dict[j])

#             string = string[:idx]+string[idx+w_len:]

#             j = 0
        
#         else:
#             j += 1
    
    
#     return False if string else True

# print(workBreak(string=s, word_dict=wordDict))

'This solution goes up to the 74% of the test cases'

# Dynamic Programming Approach

# def workBreak(string:str, word_dict:list[str]) -> bool:

#     dp = [False] * (len(s) + 1) # dp[i] means s[:i+1] can be segmented into words in the wordDicts 
#     dp[0] = True

#     for i in range(len(s)):

#         for j in range(i, len(s)):
            
#             i_dp = dp[i]
#             sub_s = s[i: j+1]
#             test = sub_s in wordDict

#             if i_dp and test:
#                 dp[j+1] = True
                
#     return dp[-1]

# print(workBreak(string=s, word_dict=wordDict))




'''140. Word Break II'''

#Input

# #Case 1
# s = "catsanddog"
# wordDict = ["cat","cats","and","sand","dog"]
# #Output: ["cats and dog","cat sand dog"]

# #Case 2
# s = "pineapplepenapple"
# wordDict = ["apple","pen","applepen","pine","pineapple"]
# #Output: ["pine apple pen apple","pineapple pen apple","pine applepen apple"]

# #Case 3
# s = "catsandog"
# wordDict = ["cats","dog","sand","and","cat"]
# #Output: []


# My Approach

'''
Intuition:

    - With the solution of the last exercise, bring the found words into a list and join them to from a sentence.
    - In a loop, check if the first found word is the same of the last sentece, if do, keep searching for another word,
        - if not found words after looping from the first character, end the loop.
'''

# def wordBreak(s:str, wordDict:list[str]) -> list[str]:

#     sentences = []
#     sent = []
#     string = s
#     lasts_first_word = []

#     while True:

#         j = 0

#         while j < len(string):

#             if string[0:j+1] in wordDict and string[0:j+1] not in lasts_first_word:

#                 sent.append(string[0:j+1])
#                 string = string[j+1:]
#                 j = 0
            
#             else:
#                 j += 1
        

#         if sent:
#             sentences.append(' '.join(sent))
#             string = s
#             lasts_first_word.append(sent[0])
#             sent = []
        
#         else:
#             break
    
#     return sentences        

# print(wordBreak(s=s, wordDict=wordDict))

"This solution doesn't even get to pass all the initial test cases, but at least it worked as a challenge to do at least one"


# Backtracking & Recursion approach

# def wordBreakHelper(s:str, start:int, word_set:set, memo:dict) -> list[str]:

#     if start in memo:
#         return memo[start]
    
#     valid_substr = []

#     if start == len(s):
#         valid_substr.append('')

#     for end in range(start+1, len(s)+1):

#         prefix = s[start:end]

#         if prefix in word_set:

#             suffixes = wordBreakHelper(s, end, word_set, memo)

#             for suffix in suffixes:

#                 valid_substr.append(prefix + ('' if suffix == '' else ' ') + suffix)

#     memo[start] = valid_substr

#     return valid_substr
         

# def wordBreak(s:str, wordDict: list[str]) -> list[str]:

#     memo = {}
#     word_set = set(wordDict)
#     return wordBreakHelper(s, 0, word_set, memo)


# print(wordBreak(s=s, wordDict=wordDict))




'''141. Linked List Cycle'''

# # Base
# class ListNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.next = None


# Input

# # Case 1
# head_layout = [3,2,0,-4]

# head = ListNode(x=3)
# pos1 = ListNode(x=2)
# pos2 = ListNode(x=0)
# pos3 = ListNode(x=-4)

# head.next, pos1.next, pos2.next, pos3.next = pos1, pos2, pos3, pos1
# # Output: True / Pos1

# # Case 2
# head_layout = [1,2]

# head = ListNode(x=1)
# pos1 = ListNode(x=2)

# head.next, pos1.next = pos1, head
# # Output: True / Pos0

# # Case 3
# head_layout = [1]

# head = ListNode(x=1)
# # Output: False / pos-1


# def hasCycle(head:ListNode) -> bool:

#     if head is None or head.next == None:
#         return False
    

#     visited = []

#     curr = head

#     while curr is not None:

#         if curr in visited:
#             return True
        
#         visited.append(curr)

#         curr = curr.next
    
#     return False

# print(hasCycle(head=head))


'This a suboptimal solution, it works but it takes considerable memory to solve it'

# Another approach (Probing)

'''
Explanation
    By making two markers initialized in the head one with the double of the "speed" of the other, if those are in a cycle
    at some point they got to meet, it means there is a cycle in the list, but if one if the faster gets to None,
    that'll mean that there is no cycle in there.
'''

# def hasCycle(head:ListNode) -> bool:

#     if not head:
#         return False
    
#     slow = fast = head

#     while fast and fast.next:

#         slow = slow.next
#         fast = fast.next.next

#         if slow == fast:
#             return True
    
#     return False

# print(hasCycle(head=head))




'''146. LRU Cache'''

# Input
# commands = ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
# inputs = [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]

# Output: [null, null, null, 1, null, -1, null, -1, 3, 4]


# My Approach

'''
Intuition

    The use of 'OrderedDicts' from the Collections module will be useful to keep track
    of the last recently used values
'''


# class LRUCache(object):   

#     def __init__(self, capacity):
#         """
#         :type capacity: int
#         """     

#         self.capacity = capacity
#         self.capacity_count = 0
#         self.memory = {}
        

#     def get(self, key):
#         """
#         :type key: int
#         :rtype: int
#         """

#         output = self.memory.get(key,-1)

#         if output != -1:

#             item = (key, self.memory[key])
#             del self.memory[item[0]]
#             self.memory[item[0]] = item[1]

#         return output
        

#     def put(self, key, value):
#         """
#         :type key: int
#         :type value: int
#         :rtype: None
#         """

#         existing_key = self.memory.get(key, -1)

#         if existing_key == -1:
#             self.memory[key] = value

#         else:
#             self.memory.update({key:value})

#             item = (key, value)
#             del self.memory[item[0]]
#             self.memory[item[0]] = item[1]
        
#         self.capacity_count += 1

#         if self.capacity_count > self.capacity:

#             del_item = list(self.memory.keys())[0]
#             del self.memory[del_item]
            
#             self.capacity_count = self.capacity

        


# Your LRUCache object will be instantiated and called as such:
# obj = LRUCache(capacity)
# param_1 = obj.get(key)
# obj.put(key,value)


# a = {'a':1, 'b':2, 'c':3}

# print(a)

# item = ('a', a['a'])

# del a[item[0]]

# a[item[0]] = item[1]


# print(a)




'''148. Sort List'''

# Base
# class ListNode(object):
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next


# Input

# # Case 1
# list_layout = [4,2,1,3]
# head = ListNode(val=4, next=ListNode(val=2, next=ListNode(val=1, next=ListNode(val=3))))
# # Output: [1,2,3,4]

# # Case 2
# list_layout = [-1,5,3,4,0]
# head = ListNode(val=-1, next=ListNode(val=5, next=ListNode(val=3, next=ListNode(val=4, next=ListNode(val=0)))))
# # Output: [-1,0,3,4,5]

# # Case 3
# list_layout = [1,2,3,4]
# head = ListNode(val=1, next=ListNode(val=2, next=ListNode(val=3, next=ListNode(val=4))))
# # Output: [1,2,3,4]


# My Approach

'''
Intuition

    - Brute force: Traverse the list to collect each node with its value in a list,
    and apply some sorting algorithm to sort them.

'''

# def sortList(head):

#     if not head:
#         return ListNode()
    
#     curr = head
#     holder = []

#     while curr:

#         holder.append([curr.val, curr])
#         curr = curr.next


#     def merge_sort(li):

#         if len(li)<=1:
#             return li
        
#         left_side = li[:len(li)//2]
#         right_side = li[len(li)//2:]

#         left_side = merge_sort(left_side)
#         right_side = merge_sort(right_side)

#         return merge(left=left_side, right=right_side)


#     def merge(left, right):
        
#         i = j = 0
#         result = []

#         while i < len(left) and j < len(right):

#             if left[i][0] < right[j][0]:
#                 result.append(left[i])
#                 i+=1
            
#             else:
#                 result.append(right[j])
#                 j+=1

#         while i < len(left):
#             result.append(left[i])
#             i+=1
        
#         while j < len(right):
#             result.append(right[j])
#             j+=1

#         return result

#     sorted_list = merge_sort(li=holder)
    
#     for i in range(len(sorted_list)):

#         if i == len(sorted_list)-1:
#             sorted_list[i][1].next = None
        
#         else:
#             sorted_list[i][1].next = sorted_list[i+1][1]
    
#     return sorted_list[0][1]

# test = sortList(head=head)




'''149. Max Points on a Line'''

'''
Revision

    The problem could be pretty hard if no math knowledge is acquired beforehand.
    By definition, if several points share the same 'slope' with one single point,
    it'd mean that they are all included in the same line.

    So the problem reduces to (brut force) check for each point if the rest share the same
    slope and the biggest group with common slope will be the answer
'''

# def maxPoints(points:list[list[int]]):

#     # if there is no more than a pair of point in the plane, well, that's the answer
#     if len(points) < 3:
#         return len(points)
    
#     # Initializing with the lowest possible answer
#     result = 2

#     # Since we are counting on pairs, we're iterating up to the second last point of the group
#     for i, point1 in enumerate(points[:-1]):

#         slopes = {} # The keys will be the slopes and the values the count of points with the same slope

#         for point2 in points[i+1:]:
            
#             slope = None
#             x_comp = point2[0] - point1[0]

#             if x_comp:  # The bool of 0 is False
                
#                 # Calculate the slope
#                 slope = (point2[1] - point1[1]) / x_comp

#             # If that slope already exist, add one point to the count
#             if slope in slopes:

#                 slopes[slope] += 1
#                 new = slopes[slope]

#                 result = max(result, new)
            
#             # else, create a new dict entry
#             else:
#                 slopes[slope] = 2

#     return result




'''149. Max Points on a Line'''















