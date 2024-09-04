'''
CHALLENGES INDEX

98. Validate Binary Search Tree (Tree) (DFS)
101. Symmetric Tree (Tree) (BFS) (DFS)
102. Binary Tree Level Order Traversal (Tree) (BFS) (DFS)
103. Binary Tree Zigzag Level Order Traversal (BFS) (DFS)
104. Maximum Depth of Binary Tree (Tree) (BFS) (DFS)
105. Construct Binary Tree from Preorder and Inorder Traversal (DQ) (Tree)
108. Convert Sorted Array to Binary Search Tree (DQ) (Tree)
116. Populating Next Right Pointers in Each Node (BFS) (DFS) (Tree)
124. Binary Tree Maximum Path Sum (DP) (Tree) (DFS)
207. Course Schedule (DFS) (Topological Sort)
208. Implement Trie (Hash Table) (Tree)
210. Course Schedule II (DFS) (Topological Sort)
230. Kth Smallest Element in a BST (Heap) (DFS) (Tree)


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

'105. Construct Binary Tree from Preorder and Inorder Traversal'
# def x():

#     # Base
#     class TreeNode(object):
#         def __init__(self, val=0, left=None, right=None):
#             self.val = val
#             self.left = left
#             self.right = right


#     # Input

#     # Case 1
#     preorder, inorder = [3,9,20,15,7],[9,3,15,20,7]
#     # Output: [3,9,20,None,None,15,7]


#     def buildTree(preorder, inorder):

#         if inorder:

#             idx = inorder.index(preorder.pop(0))
#             root = TreeNode(val = inorder[idx])
#             root.left = buildTree(preorder=preorder, inorder=inorder[:idx])
#             root.right = buildTree(preorder=preorder, inorder=inorder[idx+1:])

#             return root
        
#     'Done'

'108. Convert Sorted Array to Binary Search Tree'
# def x():

#     # Base
#     class TreeNode(object):
#         def __init__(self, val=0, left=None, right=None):
#             self.val = val
#             self.left = left
#             self.right = right

#     # Input
#     # Case 1
#     nums = [-10,-3,0,5,9]
#     # Output: [0,-3,9,-10,None,5] | [0,-10,5,None,-3,None,9]

#     # Case 2
#     nums = [1,3]
#     # Output: [3,1] | [1,None,-3]

#     '''
#     My Approach

#         Intuition:
#                 Learnt for the prior exercise, the middle node will be taken as the root.
#                 from there, it can recursively built the solution.
                
#                 base case = when len(nums) = 0
#     '''

#     def sortedArrayToBST(nums:list[int]) -> TreeNode:

#         nums_len = len(nums)

#         if nums_len:

#             idx = nums_len // 2

#             return TreeNode(val = nums[idx], left = sortedArrayToBST(nums=nums[:idx]), right = sortedArrayToBST(nums=nums[idx+1:]))

#     # Testing
#     node = sortedArrayToBST(nums=nums)
#     print(node)

#     'Done'

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

'''124. Binary Tree Maximum Path Sum'''
# def x():

#     # Base 
#     class TreeNode(object):
#         def __init__(self, val=0, left=None, right=None):
#             self.val = val
#             self.left = left
#             self.right = right

#     # Input
#     #Case 1
#     tree_layout = [1,2,3]
#     root = TreeNode(val=1, left=TreeNode(val=2), right=TreeNode(val=3))
#     #Output: 6

#     #Case 2
#     tree_layout = [-10,9,20,None, None,15,7]
#     left = TreeNode(val=9)
#     right = TreeNode(val=20, left=TreeNode(val=15), right=TreeNode(val=7))
#     root = TreeNode(val=-10, left=left, right=right)
#     #Output: 42

#     #Custom Case
#     tree_layout = [1,-2,3,1,-1,-2,-3]
#     left = TreeNode(val=-2, left=TreeNode(val=1), right=TreeNode(val=3))
#     right = TreeNode(val=-3, left=TreeNode(val=-2, left=TreeNode(val=-1)))
#     root = TreeNode(val=1, left=left, right=right)
#     #Output: 3


#     '''
#     My Approach

#         Intuition:
#             - Make a preorder traversal tree list.
#             - Apply Kadane's algorithm to that list.
#     '''

#     def maxPathSum(root:TreeNode) -> int:

#         #First, Preorder
#         path = []

#         def preorder(node:TreeNode) -> None:

#             if node:
#                 preorder(node=node.left)
#                 path.append(node.val)
#                 preorder(node=node.right)

#         preorder(node=root)

#         #Now Kadane's
#         max_so_far = max_end_here = path[0]

#         for num in path[1:]:

#             max_end_here = max(num, max_end_here + num)
#             max_so_far = max(max_so_far, max_end_here)

#         return max_so_far

#     # Testing
#     print(maxPathSum(root=root))

#     '''
#     Notes:
#         - On the first run it went up to 59% of the cases, thats Kudos for me! :D
#         - The problem with this algorithm is that it supposes that after reaching a parent and child node,
#         it's possible to go from a right child to the parent of the parent and that either forcibly makes
#         to pass twice from the parent before going to the granparent, or that one grandchild is connected
#         to the grandfather, which is also out of the rules.

#         I misinterpret this because one of the examples showed a path [leftchild, parent, rightchild] which
#         is valid only if we don't want to pass thruough the grandparent.
        
#         The best choice here is to make a recursive proning algorithm
#     '''


#     'A recursive approach'
#     def maxPathSum(root):

#         max_path = float('-inf') #Placeholder

#         def get_max_gain(node):

#             nonlocal max_path

#             if not node:
#                 return 0
            
#             gain_on_left = max(get_max_gain(node.left),0)
#             gain_on_right = max(get_max_gain(node.right),0)

#             current_max_path = node.val + gain_on_left + gain_on_right
#             max_path = max(max_path, current_max_path)

#             return node.val + max(gain_on_left, gain_on_right)
        
#         get_max_gain(root)

#         return max_path

#     # Testing
#     print(maxPathSum(root))

#     'Done'

'''207. Course Schedule'''
# def x():

#     # Input
#     # Case 1
#     numCourses = 2
#     prerequisites = [[1,0]]
#     # Output: True

#     # Case 2
#     numCurses = 2
#     prerequisites = [[1,0], [0,1]]
#     # Output: False


#     'DFS Approach'
#     def canFinish(numCourses:int, prerequisites: list[list[int]]) -> bool:

#         # Create the graph
#         preMap = {course:[] for course in range(numCourses)}

#         # Populate the graph
#         for crs, pre in prerequisites:
#             preMap[crs].append(pre)

#         # Create a visit (set) to check the current branch visited (to detect cycles)
#         visit_set = set()

#         # Define the DFS func
#         def dfs(node):

#             # Base case where is a cylce
#             if node in visit_set:
#                 return False
            
#             # Base case where not prerequisites
#             if preMap[node] == []:
#                 return True
            
#             visit_set.add(node)

#             for prereq in preMap[node]:
                
#                 if not dfs(prereq):
#                     return False

#             visit_set.remove(node)
#             preMap[prereq] = [] # As it passes, then cleared the list in case is a prereq of something else
#             return True
        
#         courses = sorted(set(x for pair in prerequisites for x in pair))

#         for crs in courses:        
#             if not dfs(crs):
#                 return False
        
#         return True

#     # Testing
#     print(canFinish(numCourses, prerequisites))

#     'Done'

'''208. Implement Trie (Prefix Tree)'''
# def x():

#     # Implementation
#     class TrieNode:

#         def __init__(self, is_word=False):
#             self.values = {}
#             self.is_word = is_word

#     'Solution'
#     class Trie:

#         def __init__(self):
#             self.root = TrieNode()
    

#         def insert(self, word: str) -> None:

#             node = self.root

#             for char in word:

#                 if char not in node.values:
#                     node.values[char] = TrieNode()
                
#                 node = node.values[char]

#             node.is_word = True


#         def search(self, word: str) -> bool:
            
#             node = self.root

#             for char in word:          
                        
#                 if char not in node.values:
#                     return False
                
#                 node = node.values[char]
            
#             return node.is_word


#         def startsWith(self, prefix: str) -> bool:
            
#             node = self.root

#             for char in prefix:

#                 if char not in node.values:
#                     return False
                
#                 node = node.values[char]
            
#             return True

#     # Testing
#     new_trie = Trie()
#     new_trie.insert('Carrot')
#     print(new_trie.startsWith('Car'))  

#     'Done'

'''210. Course Schedule II'''
# def x():

#     # Input
#     # Case 1
#     numCourses = 2
#     prerequisites = [[0,1]]
#     # Output: True

#     # Case 2
#     numCourses = 4
#     prerequisites = [[1,0],[2,0],[3,1],[3,2]]
#     # Output: [0,1,2,3] or [0,2,1,3]

#     # Case 3
#     numCourses = 1
#     prerequisites = []
#     # Output: [0]

#     # Custom Case
#     numCourses = 3
#     prerequisites = [[1,0]]
#     # Output: [0]


#     'My approach'
#     def findOrder(numCourses:int, prerequisites: list[list[int]]) -> list[int]:

#         # Handling corner case
#         if not prerequisites:
#             return [x for x in range(numCourses)]
        
#         # Create the graph as an Adjacency list
#         pre_map = {course:[] for course in range(numCourses)}

#         # Populate the graph
#         for crs, pre in prerequisites:
#             pre_map[crs].append(pre)

#         # Create the visit set to watch for cycles
#         visit_set = set()

#         # Create the path in which the order of the courses will be stored
#         path = []

#         # Define the recursive dfs func
#         def dfs(course):

#             # If we get to a course we already pass through, means we're in a Cycle
#             if course in visit_set:
#                 return False

#             # If we get to a course that has no prerequisites, means we can take it
#             if pre_map[course] == []:

#                 path.append(course) if course not in path else None

#                 return True
            
#             visit_set.add(course)   # Mark the course as visited

#             # Check if the course's prerequisites are available to take
#             for prereq in pre_map[course]:
                
#                 if dfs(prereq) is False:
#                     return False
                
#             visit_set.remove(course)
#             pre_map[course] = []
#             path.append(course)  # Build the path backwards

#             return True


#         # # Create a list with all the courses available
#         # courses = sorted(set(x for pair in prerequisites for x in pair))


#         # Run through all the courses
#         for crs in range(numCourses):
#             if dfs(crs) is False:
#                 return []
            
#         return path

#     # Testing
#     print(findOrder(numCourses=numCourses, prerequisites=prerequisites))

#     'Note: It worked based on the first case version'

'''230. Kth Smallest Element in a BST'''
# def x():

#     # Base
#     # Definition for a binary tree node.
#     class TreeNode:
#         def __init__(self, val=0, left=None, right=None):
#             self.val = val
#             self.left = left
#             self.right = right

#     # Case 1
#     tree_layout = [3,1,4,None,2]
#     one, four = TreeNode(val=1, right=TreeNode(val=2)), TreeNode(val=4)
#     root = TreeNode(val=3, left=one, right=four)
#     k = 1
#     # Output: 1

#     # Case 2
#     tree_layout = [5,3,6,2,4,None,None,1]
#     three, six = TreeNode(val=3, left=TreeNode(val=2, left=TreeNode(val=1)), right=TreeNode(val=4)), TreeNode(val=6)
#     root = TreeNode(val=5, left=three, right=six)
#     k = 3
#     # Output: 3

#     # Custom Case
#     tree_layout = [5,3,6,2,4,None,None,1]
#     three, six = TreeNode(val=3, left=TreeNode(val=2, left=TreeNode(val=1)), right=TreeNode(val=4)), TreeNode(val=6)
#     root = TreeNode(val=5, left=three, right=six)
#     k = 3
#     # Output: 3


#     '''
#     My Aprroach
   
#         Intuition:
#             - Traverse the Tree with preorder to extract the values
#             - Create a Max heap of length k and go through the rest of the elements (mantaining the heap property).
#             - Return the first element of the heap.
#     '''

#     def kth_smallest(root: TreeNode,k: int) -> int:

#         # Define Aux Inorder traversal func
#         def inorder(root: TreeNode, path:list) -> list:

#             if root:

#                 node = root

#                 inorder(root=node.left, path=path)
#                 path.append(node.val)
#                 inorder(root=node.right, path=path)

#                 return path

#         tree_list = inorder(root=root, path=[])

#         tree_list.sort()

#         return tree_list[k-1]

#     # Testing
#     print(kth_smallest(root=root, k=k))

#     '''Notes: 
#     - This approach works perfectly, and it beated 37% of solutions in Runtime and 80% in space.
        
#         Complexity:
#         - Time complexity: O(nlogn).
#         - Space Complexity: O(n).

#     Now, if no sorting func is required to be used, below will be that version.
#     '''


#     'Without Sorting Approach'
#     import heapq

#     def kth_smallest(root: TreeNode,k: int) -> int:

#         # Define Aux Inorder traversal func
#         def inorder(root: TreeNode, path:list) -> list:

#             if root:

#                 node = root

#                 inorder(root=node.left, path=path)
#                 path.append(node.val)
#                 inorder(root=node.right, path=path)

#                 return path

#         # Extract the tree nodes values in a list
#         tree_list = inorder(root=root, path=[])


#         # Make a min-heap out of the tree_list up to the 'k' limit
#         heap = tree_list[:k]
#         heapq.heapify(heap)

#         # Iterate through each element in the tree_list starting from 'k' up to len(tree_list)
#         for num in tree_list[k:]:

#             if num < heap[0]:
#                 heapq.heappop(heap)
#                 heapq.heappush(heap, num)
        
#         return heap[-1] # The result is the last element of the min-heap, since it was length k, and the last is the kth

#     # Testing
#     print(kth_smallest(root=root, k=k))

#     '''Notes: 
#     - This approach also worked smoothly, and it consequentially reduced its performance
#         beating only 6% of solutions in Runtime and it maintains the 80% in space.
        
#         Complexity:
#         - Time complexity: O(n+(n-k)logk).
#         - Space Complexity: O(n).

#     Now, what if I don't traverse the elements (O(n)) and later I traverse up to k?
#         Would it be possible to order the heap while traversing the tree?.
#     '''

#     'Another enhanced solution'
#     import heapq

#     def kth_smallest(root: TreeNode, k: int) -> int:

#         # Define the heap with 'inf' as it first element (To be pushed later on)
#         heap = [float('inf')]

#         # Define Aux Inorder traversal func
#         def inorder(root: TreeNode) -> None:

#             if root:

#                 node = root

#                 inorder(root=node.left)

#                 if len(heap) == k:

#                     if node.val < heap[0]:
#                         heapq.heappop(heap)
#                         heapq.heappush(heap, node.val)
#                         pass
                
#                 else:
#                     heap.append(node.val)


#                 inorder(root=node.right)
        
#         inorder(root=root)
        
#         return heap[-1] # The result is the last element of the min-heap, since it was length k, and the last is the kth

#     # Testing
#     print(kth_smallest(root=root, k=k))

#     '''Notes: 
#     - This approach also worked smoothly, and it actually beated the first approach in performance,
#         beating 57% of solutions in Runtime and it maintains the 80% in space.
        
#         Complexity:
#         - Time complexity: O(nlogk).
#         - Space Complexity: O(n+k).

#     That was a great exercise, now what is the customary solution for this?.
#         Quick answer: Simply inorderlt traverse the tree up to k, since is a Binary Search Tree, it was already sorted.
#     '''

#     'Done'













































