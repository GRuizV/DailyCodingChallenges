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






'''xxx'''