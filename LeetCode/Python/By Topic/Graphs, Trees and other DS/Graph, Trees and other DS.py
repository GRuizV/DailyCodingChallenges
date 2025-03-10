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
297. Serialize and Deserialize Binary Tree (BFS) (Tree)

114. Flatten Binary Tree to Linked List (RC) (Tree)
199. Binary Tree Right Side View (Tree) (DFS) (RC)
226. Invert Binary Tree (Tree) (DFS)
543. Diameter of Binary Tree (Tree)



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

(18)
'''

# Base Definition of TreeNode & Tree Print Func
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def pretty_print_bst(node:TreeNode, prefix="", is_left=True):

    if not node:
        return

    if node.right is not None:
        pretty_print_bst(node.right, prefix + ("│   " if is_left else "    "), False)

    print(prefix + ("└── " if is_left else "┌── ") + str(node.val))

    if node.left is not None:
        pretty_print_bst(node.left, prefix + ("    " if is_left else "│   "), True)


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

'''297. Serialize and Deserialize Binary Tree'''
# def x():

#     # Definition for a binary tree node.
#     class TreeNode(object):
#         def __init__(self, x):
#             self.val = x
#             self.left = None
#             self.right = None


#     # Input
#     # Case 1
#     root_map = [1,2,3,None,None,4,5]
#     root = TreeNode(1)
#     two, three = TreeNode(2), TreeNode(3)
#     four, five = TreeNode(4), TreeNode(5)
#     root.left, root.right = two, three
#     three.left, three.right = four, five

#     # Custom Case
#     root_map = [4,-7,-3,None,None,-9,-3,9,-7,-4,None,6,None,-6,-6,None,None,0,6,5,None,9,None,None,-1,-4,None,None,None,-2]
#     root = TreeNode(4)
#     two, three = TreeNode(-7), TreeNode(-3)
#     root.left, root.right = two, three
#     four, five = TreeNode(-9), TreeNode(-3)
#     three.left, three.right = four, five
#     six, seven, eight = TreeNode(9), TreeNode(-7), TreeNode(-4)
#     four.left, four.right = six, seven
#     five.left = eight
#     nine, ten = TreeNode(6), TreeNode(-6)
#     seven.left = nine
#     eight.right = ten
#     eleven, twelve, thirteen = TreeNode(-6), TreeNode(0), TreeNode(6)
#     nine.left, nine.right = eleven, twelve
#     ten.left = thirteen
#     fourteen, fifteen = TreeNode(5), TreeNode(-2)
#     thirteen.left, thirteen.right = fourteen, fifteen
#     sixteen, seventeen = TreeNode(9), TreeNode(-1)
#     fourteen.left, fourteen.right = sixteen, seventeen
#     eighteen = TreeNode(-4)
#     seventeen.left = eighteen
#     # Output: [4,-7,-3,null,null,-9,-3,9,-7,-4,null,6,null,-6,-6,null,null,0,6,5,null,9,null,null,-1,-4,null,null,null,-2]


#     'Solution'
#     class Codec:

#         def serialize(self, root):
#             """
#             Encodes a tree to a single string.
            
#             :type root: TreeNode
#             :rtype: str
#             """

#             # Handle corner case
#             if not root:
#                 return ''

#             queue = [root]
#             visited = []

#             while queue:

#                 node = queue.pop(0)

#                 if node:
#                     visited.append(str(node.val))
#                     queue.extend([node.left, node.right])

#                 else:
#                     visited.append('None')        

#             return ','.join(visited)
            

#         def deserialize(self, data):
#             """
#             Decodes your encoded data to tree.
            
#             :type data: str
#             :rtype: TreeNode
#             """

#             # Handle corner case
#             if not data:
#                 return
            
#             # Transform data into a valid input for the tree
#             data = [int(x) if x != 'None' else None for x in data.split(',')]

#             # Initilize the root
#             root = TreeNode(data[0])

#             # Populate the tree
#             index = 1
#             queue = [root]

#             while index < len(data) and queue:

#                 node = queue.pop(0)

#                 if data[index]:

#                     node.left = TreeNode(data[index])
#                     queue.append(node.left)
                
#                 index += 1
                
#                 if data[index]:

#                     node.right = TreeNode(data[index])
#                     queue.append(node.right)
                
#                 index += 1

#             return root


#     # Testing
#     ser = Codec()
#     deser = Codec()
#     ans = deser.serialize(root=root)

#     # Your Codec object will be instantiated and called as such:
#     ser = Codec()
#     deser = Codec()
#     ans = deser.deserialize(ser.serialize(root))

#     # Auxiliary pretty print function
#     def pretty_print_bst(node, prefix="", is_left=True):

#         if not node:
#             node = root
        
#         if not node:
#             print('Empty Tree')
#             return


#         if node.right is not None:
#             pretty_print_bst(node.right, prefix + ("│   " if is_left else "    "), False)

#         print(prefix + ("└── " if is_left else "┌── ") + str(node.val))

#         if node.left is not None:
#             pretty_print_bst(node.left, prefix + ("    " if is_left else "│   "), True)


#     pretty_print_bst(node=root)
#     print('\n\n\n\n')
#     pretty_print_bst(node=ans)

#     'Done'




'''114. Flatten Binary Tree to Linked List'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     root = [1,2,5,3,4,None,6]
#     root = TreeNode(1)
#     root.left = TreeNode(2)
#     root.right = TreeNode(5)
#     root.left.left = TreeNode(3)
#     root.left.right = TreeNode(4)
#     root.right.right = TreeNode(6)
#     # Output: [1,None,2,None,3,None,4,None,5,None,6]

#     # Case 2
#     root = []
#     # Output: []

#     # Case 3
#     root = [0]
#     root = TreeNode(0)
#     # Output: [0]

#     # Case 4
#     root = [1,2,3]
#     root = TreeNode(1)
#     root.left = TreeNode(2)
#     root.right = TreeNode(3)
#     # Output: [1,None,2,None,3]


#     '''
#     My Approach

#         Intuition:
            
#             - Initialize a dummy tree node holder at 0.
#             - In a preorder traversal keep assigning each visited node to the right pointer of dummy.
#             - Assign the right of dummy to root.
#     '''

#     def flatten(root: Optional[TreeNode]) -> None:

#         # Handle Corner case: if not root
#         if not root:
#             return None
        
#         # Initialize the dummy holder
#         dummy = TreeNode(0)

#         # Initialize a list holder to store the preorder traversal result
#         visited = []

#         def preorder(node: TreeNode) -> None:

#             if node:
#                 visited.append(node) 
#                 preorder(node=node.left)
#                 preorder(node=node.right)


#         # Call the preorder func
#         preorder(node=root)

#         # Initialize a TreeNode holder to help the traverse
#         curr = TreeNode(0)

#         # Assign the right of dummy to curr
#         dummy.right = curr

#         # Traverse through the visited nodes
#         for node in visited:
#             node.left, node.right = None, None
#             curr.right = node
#             curr = curr.right
                
#         # Reassign root
#         root = dummy.right.right

#         # # For testing: Return the modified dummy and the visited list
#         # return dummy.right, visited

#     # Testing
#     node, li = flatten(root=root)

#     print([elem.val for elem in li])

#     '''Note: While this way works, is inefficient, its time complexity goes up to O(n^2)'''




#     """A More Efficient Approach"""

#     def flatten(root: Optional[TreeNode]) -> None:

#         # Handle Corner case: if not root
#         if not root:
#             return None
        
#         # Initialize a mutable object to be modified inside the preorder func
#         node = [None]

#         def preorder(node: TreeNode) -> None:
            
#             # Node existing guard
#             if not node:
#                 return None

#             # If the node exist, flatten it
#             if node[0]:
                
#                 node[0].left = None
#                 node[0].right = node # Link it to the current node

#             # Move curr to the current node
#             node[0] = node

#             # Save the right subtree for the other recursive calls
#             right_subtree = node.right
            
#             preorder(node=node.left)    # Move to the left part of the tree first (Preorder)
#             preorder(node=right_subtree)    # Move to the right part of the tree after

#         # Call the preorder func
#         preorder(node=root)

#     '''Note: This approach only goes up to O(n)'''

'''199. Binary Tree Right Side View'''
# def x():
    
#     from typing import Optional

#     # Definition for a binary tree node.
#     class TreeNode:
#         def __init__(self, val=0, left=None, right=None):
#             self.val = val
#             self.left = left
#             self.right = right

#     # Input
#     # Case 1
#     tree = [1,2,3,None,5,None,4]
#     root = TreeNode(val=1,
#                     left=TreeNode(val=2,
#                                   right=TreeNode(val=5)),
#                     right=TreeNode(val=3,
#                                    right=TreeNode(val=4))                    
#                     )
#     # Output: [1,3,4]

#     # Case 2
#     tree = [1,None,3]
#     root = TreeNode(val=1, right=TreeNode(val=3))
#     # Output: [1,3]

#     # Case 3
#     tree = []
#     root = None
#     # Output: []

#     # Custom Case
#     tree = [1,2,3,4]
#     root = TreeNode(val=1,
#                     left=TreeNode(val=2,
#                                   left=TreeNode(val=4)),
#                     right=TreeNode(val=3)
#                     )
#     # Output: [1,2]

#     '''
#     My Approach (DFS)

#         Intuition:
            
#             - Handle corner case: No Input, return an empty list.
#             - Create a nodes values holder named 'result' to be returned once the tree is processed.
#             - Create a nodes holder named 'stack' and with the root node as its only element.
#             - In a while loop - whit condition while 'stack exists':
#                 * Add the value of the node to 'result'.
#                 * Add the current node right pointer content to 'stack' if there is one.
#                     + Otherwhise: add the left node to the stack.
#             - Return 'result'.
#     '''

#     def rightSideView(root: Optional[TreeNode]) -> list[int]:

#         # Handle Corner case: return an empty list if no root is passed
#         if not root:
#             return []
        
#         # Create a nodes values holder named 'result'
#         result = []

#         # Create a nodes holder named 'stack'
#         stack = [root]

#         # Process the Tree
#         while stack:

#             # Pop the last element contained in the stack
#             node = stack.pop()

#             if node:

#                 # Add the value of the node to 'result'
#                 result.append(node.val)

#                 # Add the current node right pointer content to 'stack' if there is one, Otherwhise add the left node to the stack
#                 stack.append(node.right) if node.right else stack.append(node.left)
        
#         # Return 'result'
#         return result

#     # Testing
#     # print(rightSideView(root=root))

#     '''Note: This approach met 73% of the test cases'''




#     'Recursive Approach'
#     def rightSideView(root: Optional[TreeNode]) -> list[int]:

#         # Handle Corner case: return an empty list if no root is passed
#         if not root:
#             return []
        
#         # Create a nodes values holder named 'result'
#         result = []

#         # Define the recursive DFS function
#         def dfs(node:TreeNode, depth:int) -> None:

#             # Base case
#             if not node:
#                 return
            
#             # If is the first time we visit this level, add the node's value
#             if depth == len(result):
#                 result.append(node.val)

#             # Recursively call the dfs on the right side of the subtree to prioritize the right part of it
#             dfs(node=node.right, depth=depth+1)

#             # Recursively call the dfs on the left side of the subtree to make sure all level are visited.
#             dfs(node=node.left, depth=depth+1)

#         # Run the function
#         dfs(node=root, depth=0)

#         # Return 'result'
#         return result

#     # Testing
#     print(rightSideView(root=root))

#     'Note: Done!'

'''226. Invert Binary Tree'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     root = [4,2,7,1,3,6,9]
#     root = TreeNode(val=4,
#                     left=TreeNode(val=2,
#                         left=TreeNode(val=1),
#                         right=TreeNode(val=3)),                    
#                     right=TreeNode(val=7,
#                         left=TreeNode(val=6),
#                         right=TreeNode(val=9))
#                     )
#     # Output: [4,7,2,9,6,3,1]

#     # Case 2
#     root = [2,1,3]
#     root = TreeNode(val=2,
#                     left=TreeNode(val=1),                    
#                     right=TreeNode(val=3)
#                     )
#     # Output: [2,3,1]

#     '''
#     My Approach (DFS)

#         Intuition:
            
#             - Handle corner case: If no node is passed, return None.
#             - Create a stack with the root node as its only value.
#             - In a While loop, while stack exists:
                
#                 + Create a 'node' holder to receive the return of stack.pop()
#                 + If 'node' is not none:

#                     * Create a 'next_rsubtree' holder to save the left side of the tree that will go in the right pointer of 'node'.
#                     * Reassign the 'left' pointer of 'node' to hold the 'node.right' content.
#                     * Reassign the 'right' pointer of 'node' to hold the 'next_rsubtree' content.
#                     * Append node.left, node.right into 'stack'.
            
#             - Return 'root'

            
#     '''

#     def invertTree(root: Optional[TreeNode]) -> Optional[TreeNode]:

#         # Handle Corner case: If no node is passed, return None
#         if not root:
#             return None
        
#         # Create a stack with the root node as its only value
#         stack = [root]

#         # Process the tree
#         while stack:

#             #  Create a 'node' holder to receive the stack popped item
#             node = stack.pop()

#             # If 'node' is not none
#             if node:
                
#                 # Create a 'next_rsubtree' holder to save the left side of the tree that will go in the right pointer of 'node'
#                 next_rsubtree = node.left

#                 # Reassign the 'left' pointer of 'node' to hold the 'node.right' content
#                 node.left = node.right

#                 # Reassign the 'right' pointer of 'node' to hold the 'next_rsubtree' content
#                 node.right = next_rsubtree

#                 # Append node.left, node.right into 'stack'
#                 stack.extend([node.left, node.right])
        
#         # Return root
#         return root

#     # Testing
    
#     print(pretty_print_bst(node=root),end="\n\n\n")
#     print(pretty_print_bst(node=invertTree(root=root)))

#     '''Note: Done'''

'''543. Diameter of Binary Tree'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     root = [1,2,3,4,5]
#     root = TreeNode(val=1,
#                         left=TreeNode(val=2,
#                             left=TreeNode(val=4),
#                             right=TreeNode(val=5)),
#                         right=TreeNode(val=3)
#     )
#     # Output: 3 // 3 is the length of the path [4,2,1,3] or [5,2,1,3]

#     # # Case 2
#     # root = [1,2]
#     # root = TreeNode(val=1, left=TreeNode(val=2))
#     # # Output: 1


#     '''
#     Soluion (Global Diameter)

#         Explanation:
            
#             1. Global Variable:

#                 - We use a list global_diameter = [0] to store the maximum diameter found. This allows us to modify it from within the recursive height function.
                
#             2. Recursive height Function:

#                 - For each node, we recursively compute the height of its left and right subtrees.
#                 - After computing the left and right heights, we update the global diameter as 'left_height + right_height'.
#                     This is because the longest path that passes through this node would go down its left and right children.
                
#             3. Return the Height:

#                 - The height function returns 1 + max(left_height, right_height), which is the height of the node itself plus the maximum height of its children.
                
#             4. Final Output:

#                 After the recursive traversal, the global_diameter contains the maximum diameter found in the tree.
#     '''

#     def diameterOfBinaryTree(root: Optional[TreeNode]) -> int:

#         # Initiliaze the global diameter contained in a list
#         global_diameter = [0]

#         # Define the recursive 'height' function
#         def height(node:TreeNode) -> int:
            
#             if not node:
#                 return 0
            
#             # Calculate the height of both left and right subtree
#             left_height = height(node=node.left)
#             right_height = height(node=node.right)

#             # Update the global diamenter
#             global_diameter[0] = max(global_diameter[0], left_height + right_height)

#             # Return the current node height
#             return 1 + max(left_height, right_height)

#         # Call the height function onto the input
#         height(node=root)
                
#         # Return the global diameter
#         return global_diameter[0]

#     # Testing
#     print(diameterOfBinaryTree(root=root))

#     '''Note: Done'''












































