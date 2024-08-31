'''
CHALLENGES INDEX

98. Validate Binary Search Tree (Tree) (DFS)
101. Symmetric Tree (Tree) (BFS) (DFS)
102. Binary Tree Level Order Traversal (Tree) (BFS) (DFS)
103. Binary Tree Zigzag Level Order Traversal (BFS) (DFS)
104. Maximum Depth of Binary Tree (Tree) (BFS) (DFS)
116. Populating Next Right Pointers in Each Node (BFS) (DFS) (Tree)


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


'98. Validate Binary Search Tree' 
# def x():

#     # Base
#     class TreeNode(object):
#         def __init__(self, val=0, left=None, right=None):
#             self.val = val
#             self.left = left
#             self.right = right

#     # Input
#     # Case 1
#     root_layout = [2,1,3]
#     root = TreeNode(val=2, left=TreeNode(val=1), right=TreeNode(val=3))
#     # Output: True

#     # Case 2
#     root_layout  = [5,1,4,None, None, 3, 6]
#     left = TreeNode(val=1)
#     right = TreeNode(val=4, left=TreeNode(val=3), right=TreeNode(val=6)) 
#     root = TreeNode(val=5, left=left, right=right)
#     # Output: False

#     # Custom Case 1
#     root_layout  = [4,2,5,1,8,5,9,3,10,2,15]
#     root = TreeNode(val=4)
#     first_left, first_right = TreeNode(val=2), TreeNode(val=5)
#     fl_left = TreeNode(val=1)
#     fl_right = TreeNode(val=8, left=TreeNode(val=5), right=TreeNode(val=9)) 
#     fr_left = TreeNode(val=3)
#     fr_right = TreeNode(val=10, left=TreeNode(val=2), right=TreeNode(val=15)) 
#     first_left.left, first_left.right = fl_left, fl_right
#     first_right.left, first_right.right = fr_left, fr_right
#     root.left, root.right = first_left, first_right
#     # Output: True

#     # Custom Case 2
#     root_layout  = [10,9,11,3,4,7,15,8,4,13,16,12,21]
#     root = TreeNode(val=10)
#     first_left, first_right = TreeNode(val=9), TreeNode(val=11)
#     fl_left = TreeNode(val=3, left=TreeNode(val=4), right=TreeNode(val=7))
#     fl_right = TreeNode(val=15)
#     fr_left = TreeNode(val=8, left=TreeNode(val=4), right=TreeNode(val=13))
#     fr_right = TreeNode(val=16, left=TreeNode(val=12), right=TreeNode(val=21)) 
#     first_left.left, first_left.right = fl_left, fl_right
#     first_right.left, first_right.right = fr_left, fr_right
#     root.left, root.right = first_left, first_right
#     # Output: False

#     # Custom Case 3
#     root_layout  = [2,2,2]
#     root = TreeNode(val=2, left=TreeNode(val=2), right=TreeNode(val=2))
#     # Output: False


#     '''
#     My Approach

#         Intuition:

#             traverse with DFS and check each (root-child) group,
#             if balanced, check the next group, else, return False.

#             if we get to the end of the tree and there were no imbalance, return True.
#     '''

#     def dfs(root:TreeNode):

#         stack = [root]

#         while stack:

#             node = stack.pop()
#             ndv = node.val

#             if node.left or node.right:

#                 if node.left:

#                     ndlv = node.left.val
                    
#                     if node.left.val > node.val:
#                        return False
                    
#                     stack.append(node.left)
                

#                 if node.right:

#                     ndrv = node.right.val
                                                    
#                     if node.right.val < node.val:
#                        return False
                    
#                     stack.append(node.right)

#                 if node.val == node.right.val and node.val == node.left.val:
#                     return False
                
#         return True

#     # Testing
#     print(dfs(root))

#     'Note: My solution works up to 78% of the cases'


#     'Inorder Tree Traversal Approach'
#     path = []
#     def inorder(root:TreeNode, route:list):

#         if root is None:
#             return
        
#         inorder(root.left, route)
#         route.append(root.val)
#         inorder(root.right, route)

#     # Testing
#     inorder(root=root, route=path)    
#     print(path)

#     'Note: The Trick here is that the inorder traversal, basically returns a sorted list if is balanced!'

'101. Symmetric Tree' 
# def x():
    
#     # Base
#     class TreeNode(object):
#         def __init__(self, val=0, left=None, right=None):
#             self.val = val
#             self.left = left
#             self.right = right

#     # Input
#     # Case 1
#     root_layout = [1,2,2,3,4,4,3]
#     root = TreeNode(val=1)
#     first_left= TreeNode(val=2, left=TreeNode(val=3), right=TreeNode(val=4))
#     first_right = TreeNode(val=2, left=TreeNode(val=4), right=TreeNode(val=3))
#     root.left, root.right = first_left, first_right
#     # Output: True

#     # Case 2
#     root_layout = [1,2,2,None,3,None,3]
#     root = TreeNode(val=1)
#     first_left= TreeNode(val=2, right=TreeNode(val=3))
#     first_right = TreeNode(val=2, right=TreeNode(val=3))
#     root.left, root.right = first_left, first_right
#     # Output: False

#     # Custom Case 1
#     root_layout = [1,2,2,2,None,2]
#     root = TreeNode(val=1)
#     first_left= TreeNode(val=2, left=TreeNode(val=2))
#     first_right = TreeNode(val=2, left=TreeNode(val=2))
#     root.left, root.right = first_left, first_right
#     # Output: False    

#     '''
#     My approach

#         Intuition:
        
#             Return a inorder-traversal list of the trees from the first left and right node,
#             and one should be the reverse of the other.

#             Handling corner cases:
#             - If only a root: True
#             - If only a root with two leaves, if the leaves are equal: True
#             - If the number of nodes is even: False
#     '''

#     def isSymetric(root:TreeNode):

#         tree_nodes = []

#         def inorder(root):

#             if root == None:
#                 return 
            
#             inorder(root.left)
#             tree_nodes.append(root.val)
#             inorder(root.right)

#         inorder(root=root)

        
#         if len(tree_nodes) == 1:
#             return True
        
#         # If there are an even number of nodes, it can be symetrical
#         if len(tree_nodes)%2 == 0:
#             return False   
        
#         if len(tree_nodes) == 3:
#             if root.left.val == root.right.val:
#                 return True

#         mid = len(tree_nodes)//2 
#         left_tree = tree_nodes[:mid]
#         right_tree = tree_nodes[mid+1:]
        
#         return left_tree == list(reversed(right_tree))

#     # Testing
#     print(isSymetric(root))

#     'Note: This solution works for cases where all node are identical, since it didnt distinguish between left and right'


#     'Recursive Approach'
#     def is_mirror(self, n1, n2):

#         if n1 is None and n2 is None:
#             return True
        
#         if (n1 is None) or (n2 is None) or (n1.val != n2.val):
#             return False

#         return self.is_mirror(n1.left, n2.right) and self.is_mirror(n1.right, n2.left)


#     def isSymmetric(self, root):

#         return self.is_mirror(n1=root.left, n2=root.right)

#     'This solution works perfectly'

'102. Binary Tree Level Order Traversal' 
# def x():
    
#     # Base
#     class TreeNode(object):
#         def __init__(self, val=0, left=None, right=None):
#             self.val = val
#             self.left = left
#             self.right = right

#     # Input
#     # Case 1
#     root_layout = [3,9,20,None,None,15,7]

#     root = TreeNode(val=3)
#     first_left= TreeNode(val=9)
#     first_right = TreeNode(val=2, left=TreeNode(val=15), right=TreeNode(val=7))

#     root.left, root.right = first_left, first_right
#     # Output: [[3],[9,20],[15,7]]

#     # Case 2
#     root_layout = [1]
#     root = TreeNode(val=1)
#     # Output: [[1]]

#     # Case 3
#     root_layout = []
#     # Output: []

#     '''
#     My Approach
        
#         Intuition:

#             With bread-first search, I can pull the values in order by levels.

#             Given that Binary tree are binary, with the powers of 2
#             it could be calculated how many nodes exist in each level.

#             and with the l = 1 + floor(log_2(n)), the number of levels can
#             be known just having the number of nodes.        
#     '''

#     from collections import deque
#     from math import floor, log2

#     def bfs(root:TreeNode):

#         queue = deque()
#         queue.append(root)

#         path = []

#         while queue:

#             node = queue.popleft()

#             if node not in path:

#                 path.append(node)

#                 if node.left:
#                     queue.append(node.left)

#                 if node.right:
#                     queue.append(node.right)

#         return [x.val for x in path]

#     # Testing
#     nodes_list = bfs(root=root)
#     n_levels = 1 + floor(log2(len(nodes_list)))
#     result = []

#     for i in range(n_levels):

#         temp = []

#         for j in range(pow(2, i)):

#             if nodes_list:
#                 temp.append(nodes_list.pop(0))
        
#         result.append(temp)
            
#     print(result)

#     'Notes: This solution works but the leetcode interpreter didnt recognized the log2 function'


#     'A Simplier Approach'
#     def levelsOrder(root:TreeNode):

#         from collections import deque
        
#         queue = deque()
#         queue.append(root)    
#         result = []

#         while queue:

#             queue_len = len(queue)
#             level = [] 
            
#             for i in range(queue_len):

#                 node = queue.popleft()

#                 if node is not None:

#                     level.append(node.val)
#                     queue.append(node.left)
#                     queue.append(node.right)

#             if level:   
#                 result.append(level)

#         return result

#     # Testing
#     print(levelsOrder(root=root))

'103. Binary Tree Zigzag Level Order Traversal' 
# def x():

#     # Base
#     class TreeNode(object):
#         def __init__(self, val=0, left=None, right=None):
#             self.val = val
#             self.left = left
#             self.right = right

#     # Input
#     # Case 1
#     root_layout = [3,9,20,None,None,15,7]
#     root = TreeNode(val=3)
#     first_left= TreeNode(val=9)
#     first_right = TreeNode(val=20, left=TreeNode(val=15), right=TreeNode(val=7))
#     root.left, root.right = first_left, first_right
#     # Output: [[3],[20,9],[15,7]]

#     # Case 2
#     root_layout = [1]
#     root = TreeNode(val=1)
#     # Output: [[1]]

#     # Case 3
#     root_layout = []
#     # Output: []


#     '''
#     My Approach

#         Notes:

#             This will go apparently the same as the level order, but in the other way arround
#             and this time is alternating depending of the level
#     '''

#     def zigzagLevelOrder(root:TreeNode) -> list[list[int]]:

#         from collections import deque

#         queue = deque()
#         queue.append(root)
#         result = []
#         level = 1

#         while queue:

#             len_q = len(queue)
#             level_nodes = []
        
#             for i in range(len_q):

#                 node = queue.popleft()

#                 if node is not None:

#                     queue.append(node.left)
#                     queue.append(node.right)
#                     level_nodes.append(node.val)

#             if len(level_nodes) != 0:

#                 if level % 2 == 0:
#                     level_nodes = list(reversed(level_nodes))
                
#                 result.append(level_nodes)
            
#             level += 1
        
#         return result

#     # Testing
#     print(zigzagLevelOrder(root=root))

#     'It worked!'

'104. Maximum Depth of Binary Tree'
# def x():

#     # Base
#     class TreeNode(object):
#         def __init__(self, val=0, left=None, right=None):
#             self.val = val
#             self.left = left
#             self.right = right

#     # Input
#     # Case 1
#     root_layout = [3,9,20,None,None,15,7]
#     root = TreeNode(val=3)
#     first_left= TreeNode(val=9)
#     first_right = TreeNode(val=20, left=TreeNode(val=15), right=TreeNode(val=7))
#     root.left, root.right = first_left, first_right
#     # Output: 3

#     # Case 2
#     root_layout = [1, None, 2]
#     root = TreeNode(val=1, right=TreeNode(val=2))
#     # Output: 2


#     '''
#     My approach

#         Notes:
#             Here could be to ways (or more) to solve it:
#                 1. Implement the BFS by level listing (like the challenges prior to this one) and count the elements of the result
#                 2. Simply list through DFS or BFS and apply l = 1 + floor(log_2(n)), to know the number of levels, but probably leetcode won't have 
#                 the log2 function in its math module, so I'll the first way.
#     '''

#     def maxDepth(root:TreeNode) -> int:

#         from collections import deque

#         queue = deque()
#         queue.append(root)
#         result = []

#         while queue:

#             queue_len = len(queue)
#             level = []

#             for _ in range(queue_len):

#                 node = queue.popleft()

#                 if node is not None:

#                     queue.append(node.left)
#                     queue.append(node.right)

#                     level.append(node.val)

#             if level:
#                 result.append(level)
        
#         return result

#     # Testing
#     print(maxDepth(root))

#     'Done!'

'116. Populating Next Right Pointers in Each Node'
# def x():

#     # Base
#     class TreeNode(object):
#         def __init__(self, val=0, left=None, right=None, next=None):
#             self.val = val
#             self.left = left
#             self.right = right
#             self.next = next

#     # Input
#     # Case 1
#     tree_lauout = [1,2,3,4,5,6,7]
#     left = TreeNode(val=2, left=TreeNode(val=4), right=TreeNode(val=5))
#     right = TreeNode(val=3, left=TreeNode(val=6), right=TreeNode(val=7))
#     root = TreeNode(val=1, left=left, right=right)
#     # Output: [1,#,2,3,#,4,5,6,7,#]    

#     '''
#     My Approach

#         Intuition:
#             This could be solved with the BFS modified to catch nodes by level,
#             and with the level picked from each loop, modify its pointers in that order 
#     '''


#     def connect(root:TreeNode) -> TreeNode:
        
#         #Start
#         queue = [root]
        
#         while queue:

#             q_len = len(queue)
#             level = []

#             for i in range(q_len):

#                 node = queue.pop(0)

#                 if node:

#                     queue.extend([node.left, node.right])
#                     level.append(node)
            
#             if level:

#                 for i in range(len(level)):

#                     if i != len(level)-1:

#                         level[i].next = level[i+1]
        
#         return root

#     'Worked right way, YAY! :D'





























