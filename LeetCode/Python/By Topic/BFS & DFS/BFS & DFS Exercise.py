'''
CHALLENGES INDEX

98. Validate Binary Search Tree (Tree) (DFS)
101. Symmetric Tree (Tree) (BFS) (DFS)
102. Binary Tree Level Order Traversal (Tree) (BFS) (DFS)
103. Binary Tree Zigzag Level Order Traversal (BFS) (DFS)
104. Maximum Depth of Binary Tree (Tree) (BFS)
116. Populating Next Right Pointers in Each Node (BFS) (DFS) (Tree)
124. Binary Tree Maximum Path Sum (DP) (Tree) (DFS)
127. Word Ladder (Hast Table) (BFS)
130. Surrounded Regions (Matrix) (BFS) (DFS)
200. Number of Islands (Matrix) (DFS)
207. Course Schedule (DFS) (Topological Sort)
210. Course Schedule II (DFS) (Topological Sort)
212. Word Search II (Array) (DFS) (BT) (Matrix)
230. Kth Smallest Element in a BST (Heap) (DFS) (Tree)
297. Serialize and Deserialize Binary Tree (BFS) (Tree)
329. Longest Increasing Path in a Matrix (Matrix) (DFS) (MEM) (RC)
341. Flatten Nested List Iterator (DFS)

114. Flatten Binary Tree to Linked List (LL) (DFS) (Tree)
199. Binary Tree Right Side View (Tree) (DFS) (RC)
226. Invert Binary Tree (Tree) (DFS)

994. Rotting Oranges (Matrix) (BFS) *DIFFERENTIAL COORDINATES

100. Same Tree (DFS) (RC)




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


(22)
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

# The base BFS algorithm
def bfs(graph, start):
    
    import collections

    visited = set()
    queue = collections.deque([start])
    
    while queue:

        node = queue.popleft()
        
        if node not in visited:

            visited.add(node)

            print(node, end=' ') # Process the node (you can replace this with your own logic)

            # Enqueue unvisited neighbors
            for neighbor in graph.get(node, []):

                if neighbor not in visited:

                    queue.extend([neighbor])
    
    print()

# The base DFS algorithm
def dfs(graph, start):

    visited = set()
    stack = [start]

    while stack:

        node = stack.pop()

        if node not in visited:

            visited.add(node)

            print(node, end=' ') # Process the node (you can replace this with your own logic)

            if node in graph:

                for neighbor in graph[node]:

                    if neighbor not in visited:

                        stack.extend(neighbor)





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

#     from typing import Optional

#     # Input
#     # Case 1
#     root = [3,9,20,None,None,15,7]
#     root = TreeNode(3)
#     root.left = TreeNode(9)
#     root.right = TreeNode(20)
#     root.right.left = TreeNode(15)
#     root.right.right = TreeNode(7)
#     # Output: 3

#     # Case 2
#     root = [1,None,2]
#     root = TreeNode(1)
#     root.right = TreeNode(2)
#     # Output: 2

#     # Case 3
#     root = []
#     root = None
#     # Output: 0

#     # Case 4
#     root = [0]
#     root = TreeNode(0)
#     # Output: 1

#     # Case 5
#     root = [1,2,3,4,5]
#     root = TreeNode(1)
#     root.left = TreeNode(2)
#     root.right = TreeNode(3)
#     root.left.left = TreeNode(4)
#     root.left.right = TreeNode(5)
#     # Output: 3

#     # Case 6
#     root = [1,2,3,4,5,None,6,7,None,None,None,None,8]
#     root = TreeNode(1)
#     root.left = TreeNode(2)
#     root.right = TreeNode(3)
#     root.left.left, root.left.right = TreeNode(4), TreeNode(5)
#     root.right.right = TreeNode(6)
#     root.left.left.left = TreeNode(7)
#     root.right.right.right = TreeNode(8)
#     # Output: 4

#     '''
#     My Approach (BFS)

#         Intuition:
            
#             Implement BFS by lvl and return the number of elements in a holder 'lvls'. 
#     '''

#     def maxDepth(root: Optional[TreeNode]) -> int:
        
#         # Handle Corner case: No node passed
#         if not root:
#             return 0

#         # Initialize a Queue holder    
#         queue = []

#         # Add the root of the tree to the queue
#         queue.append(root)

#         # Initialize a result list holder
#         result = []

#         # Process the tree
#         while queue:

#             # Capture the current queue length
#             queue_len = len(queue)

#             # Initialize a lvl list holder
#             lvl = []

#             # Process the level's nodes
#             for _ in range(queue_len):

#                 # Pop the first element present in the list
#                 node = queue.pop(0)

#                 # if it is actually a node
#                 if node:
                    
#                     # Add the children nodes to the queue
#                     queue.append(node.left)
#                     queue.append(node.right)
                    
#                     # Add the node's value to the lvl
#                     lvl.append(node.val)            

#             # if there are any nodes in the lvl, Add the lvl to the result
#             result.append(lvl) if lvl else None

#         # Return the result's length
#         return result

#     # Testing
#     pretty_print_bst(root)
#     print(f"\nResult: {len(maxDepth(root=root))}\nActual lvls: {maxDepth(root=root)}")

#     '''Note: Done'''

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

'''127. Word Ladder'''
# def x():

#     # Input
#     #Case 1
#     begin_word, end_word, word_list = 'hit', 'cog', ['hot', 'dot', 'dog', 'lot', 'log', 'cog']
#     #Output: 5

#     #Custom Case
#     begin_word, end_word, word_list = 'a', 'c', ['a', 'b', 'c']
#     #Output: 5
    

#     '''
#     My Approach

#         Intuition:
#             1. handle the corner case: the end_word not in the word_list
#             2. create an auxiliary func that check the word against the end_word: True if differ at most by 1 char, else False.
#             3. create a counter initialized in 0
#             4. start checking the begin_word and the end_word, if False sum 1 to the count, and change to the subquent word in the word_list and do the same.
#     '''

#     def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:

#         if endWord not in wordList:
#             return 0
        
#         def check(word):
#             return False if len([x for x in word if x not in endWord]) > 1 else True
        
#         if beginWord not in wordList:
#             wordList.insert(0,beginWord)
#             count = 0
        
#         else:
#             count = 1
        
#         for elem in wordList:
#             count += 1

#             if check(elem):
#                 return count     
                
#         return 0

#     # Testing
#     print(ladderLength(beginWord=begin_word, endWord=end_word, wordList=word_list))

#     'Note: This solution only went up to the 21% of the cases'


#     'BFS approach'
#     from collections import defaultdict, deque

#     def ladderLength(beginWord: str, endWord: str, wordList: list[str]) -> int:

#         if endWord not in wordList or not endWord or not beginWord or not wordList:
#             return 0

#         L = len(beginWord)
#         all_combo_dict = defaultdict(list)

#         for word in wordList:
#             for i in range(L):
#                 all_combo_dict[word[:i] + "*" + word[i+1:]].append(word) 

#         queue = deque([(beginWord, 1)])
#         visited = set()
#         visited.add(beginWord)

#         while queue:
#             current_word, level = queue.popleft()

#             for i in range(L):
#                 intermediate_word = current_word[:i] + "*" + current_word[i+1:]

#                 for word in all_combo_dict[intermediate_word]:

#                     if word == endWord:
#                         return level + 1

#                     if word not in visited:
#                         visited.add(word)
#                         queue.append((word, level + 1))
                        
#         return 0

#     'Done'

'''130. Surrounded Regions'''
# def x():

#     #Input
#     #Case 1
#     board = [
#         ["X","X","X","X"],
#         ["X","O","O","X"],
#         ["X","X","O","X"],
#         ["X","O","X","X"]
#         ]
#     # output = [
#     #     ["X","X","X","X"],
#     #     ["X","X","X","X"],
#     #     ["X","X","X","X"],
#     #     ["X","O","X","X"]
#     #     ]

#     #Case 2
#     board = [
#         ['X']
#         ]
#     # output = [
#         # ['X']
#         # ]

#     #Custom Case
#     board = [["O","O"],["O","O"]]


#     '''
#     My Approach

#         Intuition:
#             1. Check if there is any 'O' at the boarders.
#             2. Check is there is any 'O' adjacent to the one in the boarder:
#                 - If do, add them to the not-be-flipped ground and re run.
#                 - if doesn't, flip everything to 'X' and return
#             (Do this until there is no 'O' unchecked )
#     '''

#     def solve(board:list[list[str]]) -> None:

#         M = len(board)
#         N = len(board[0])

#         no_flip = []
#         all_os = []


#         # Collect all 'O's
#         for i in range(M):
#             all_os.extend((i,j) for j in range(N) if board[i][j] == 'O')
        

#         #   Check if there is a boarder 'O' within the group
#         for i in range(len(all_os)):

#             if all_os[i][0] in (0, M-1) or all_os[i][1] in (0, N-1):
#                 no_flip.append(all_os[i])


#         # Collect the 'O's near to no_flip 'O' iteratively
#         flipped = None
#         i = 0

#         while True:

#             # Condition to end the loop
#             if len(all_os) == 0 or i == len(all_os) and flipped is False:
#                 break

#             #Collecting the possibilities of an adjacent 'O'
#             adjacents = []

#             for pos in no_flip:
#                 adjacents.extend([(pos[0]-1, pos[1]), (pos[0]+1, pos[1]), (pos[0], pos[1]-1), (pos[0], pos[1]+1)])
            
#             #Check if the current element is adjacent to any no_flip 'O'
#             if all_os[i] in adjacents:
#                 no_flip.append(all_os.pop(i))
#                 flipped = True
#                 i = 0
#                 continue

#             i += 1
#             flipped = False


#         # Rewritting the board
#         #   Resetting the board to all "X"
#         for i in range(M):
#             board[i] = ["X"]*N
        
#         #   preserving the no_flip 'O's
#         for o in no_flip:
#             board[o[0]][o[1]] = 'O'

#     # Testing
#     solve(board=board)

#     'This solution met 98.2% of the cases'


#     'DFS Approach'
#     def solve(board):

#         n,m=len(board),len(board[0])
#         seen=set()

#         def is_valid(i,j):
#             return 0 <= i < n and 0<= j <m and board[i][j]=="O" and (i,j) not in seen
        
#         def is_border(i,j):
#             return i == 0 or i == n-1 or j == 0 or j == m-1
        
#         def dfs(i,j):

#             board[i][j]="y"
#             seen.add((i,j))

#             for dx , dy in ((0,1) ,(0,-1) ,(1,0),(-1,0)):
#                 new_i , new_j = dx + i , dy + j

#                 if is_valid(new_i , new_j):
#                     dfs(new_i , new_j)
            
#         for i in range(n):
#             for j in range(m):
#                 if is_border(i,j) and board[i][j]=="O":
#                     dfs(i,j) 
                    
#         for i in range(n):
#             for j in range(m):
#                 if board[i][j]=="y":
#                     board[i][j]="O"
#                 else:
#                     board[i][j]="X"

#     # Testing
#     solve(board)

#     'Done'

'''200. Number of Islands'''
# def x():

#     # Input
#     # Case 1
#     grid = [
#       ["1","1","1","1","0"],
#       ["1","1","0","1","0"],
#       ["1","1","0","0","0"],
#       ["0","0","0","0","0"]
#     ]
#     # Ouput: 1

#     # Case 2
#     grid = [
#       ["1","1","0","0","0"],
#       ["1","1","0","0","0"],
#       ["0","0","1","0","0"],
#       ["0","0","0","1","1"]
#     ]
#     # Ouput: 3

#     # Custom Case
#     grid = [
#         ["1","0"]
#         ]
#     # Ouput: 1


#     'My BFS Approach'
#     def numIslands(grid:list[list[str]]) -> int:
        
#         if len(grid) == 1:
#             return len([x for x in grid[0] if x =='1'])

#         # Create the 'lands' coordinates
#         coord = []

#         # Collecting the 'lands' coordinates
#         for i, row in enumerate(grid):
#             coord.extend((i, j) for j, value in enumerate(row) if value == '1')


#         # Create the groups holder
#         islands = []
#         used = set()


#         # BFS Definition
#         def bfs(root:tuple) -> list:

#             queue = [root]
#             curr_island = []

#             while queue:

#                 land = queue.pop(0)
#                 x, y = land[0], land[1]
                
#                 if grid[x][y] == '1' and (land not in curr_island and land not in used):

#                     curr_island.append(land)
                
#                     # Define next lands to search
#                     if x == 0:
#                         if y == 0:
#                             next_lands = [(x+1,y),(x,y+1)]
                        
#                         elif y < len(grid[0])-1:
#                             next_lands = [(x+1,y),(x,y-1),(x,y+1)]
                        
#                         else:
#                             next_lands = [(x+1,y),(x,y-1)]
                    
#                     elif x < len(grid)-1:
#                         if y == 0:
#                             next_lands = [(x-1,y),(x+1,y),(x,y+1)]
                        
#                         elif y < len(grid[0])-1:
#                             next_lands = [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]
                        
#                         else:
#                             next_lands = [(x-1,y),(x+1,y),(x,y-1)]
                    
#                     else:
#                         if y == 0:
#                             next_lands = [(x-1,y),(x,y+1)]
                        
#                         elif y < len(grid[0])-1:
#                             next_lands = [(x-1,y),(x,y-1),(x,y+1)]
                        
#                         else:
#                             next_lands = [(x-1,y),(x,y-1)]
                                    
#                     # List the next lands to visit
#                     for next_land in next_lands:

#                         if next_land not in curr_island:

#                             queue.append(next_land)

#             return curr_island
            

#         # Checking all the 1s in the grid
#         for elem in coord:

#             if elem not in used:

#                 island = bfs(elem)

#                 islands.append(island)
#                 used.update(set(island))
        
#         return len(islands)

#     # Testing
#     print(numIslands(grid=grid))
    
#     'Note: This could be done way simplier'


#     'Simplified & Corrected BFS Approach'
#     def numIslands(grid:list[list[str]]) -> int:

#         if not grid:
#             return 0

#         num_islands = 0
#         directions = [(1,0),(-1,0),(0,1),(0,-1)]

#         for i in range(len(grid)):

#             for j in range(len(grid[0])):

#                 if grid[i][j] == '1':

#                     num_islands += 1

#                     queue = [(i,j)]

#                     while queue:

#                         x, y = queue.pop(0)

#                         if 0 <= x < len(grid) and 0 <= y < len(grid[0]) and grid[x][y] == '1':

#                             grid[x][y] = '0'    # Mark as visited

#                             for dx, dy in directions:

#                                 queue.append((x + dx, y + dy))
        
#         return num_islands

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

'''212. Word Search II'''
# def x():

#     # Input
#     # Case 1
#     board = [["o","a","a","n"],["e","t","a","e"],["i","h","k","r"],["i","f","l","v"]]
#     words = ["oath","pea","eat","rain"]
#     # Output: ["eat","oath"]

#     # Case 2
#     board = [["a","b"],["c","d"]], 
#     words = ["abcb"]
#     # Output: []

#     # Custom Case
#     board = [["a","b"],["c","d"]], 
#     words = ["abcb"]
#     # Output: []


#     '''
#     My Approach
    
#         Intuiton:

#             - Based on the 'Word Seach I' backtracking solution, I will try to emulate the same but
#                 since now there are multiple word to lookout for, I will rely on a Trie implementation
#                 to look out for prefixes to optimize the process.

#                 And to try to make it work, I will pull the first letter of each word and only start
#                 the searches from those positions, so, roughly the plan is:

#                 1. Collect the coordinates of the first letter from each of the word and store them in a dict
#                     as {'word': coordinates[(x,y)]}, if a word has no coordinates and it means it won't be found
#                     in the matrix, so it won't be in Trie.
                
#                 2. Initiate the Trie with the words with coordinates.

#                 3. Iterate through each of the words, and iterate for each pair of coordinates to look out for that word,
#                     if found, add it to a result list if don't pass to the next pair of coordinates, and so on for each word.
                
#                 4. Return the found words.
#     '''

#     'ACTUAL CODE'
#     # TRIE IMPLEMENTATION

#     # TrieNode Definition
#     class TrieNode:

#         def __init__(self):
#             self.values = {}
#             self.is_word = False


#     # Trie DS Definition
#     class Trie:

#         def __init__(self):
#             self.root = TrieNode()
        
#         def insert(self, word:str) -> None:

#             curr_node = self.root

#             for char in word:

#                 if char not in curr_node.values:
#                     curr_node.values[char] = TrieNode()
                
#                 curr_node = curr_node.values[char]
            
#             curr_node.is_word = True

#         def search(self, word:str) -> bool:

#             curr_node = self.root

#             for char in word:

#                 if char not in curr_node.values:
#                     return False
                
#                 curr_node = curr_node.values[char]

#             return curr_node.is_word

#         def stars_with(self, prefix:str) -> bool:

#             curr_node = self.root

#             for char in prefix:

#                 if char not in curr_node.values:
#                     return False
                
#                 curr_node = curr_node.values[char]

#             return True

#     'Actual Solution'
#     def findWords(board: list[list[str]], words: list[str]) -> list[str]:

#         import copy

#         #AUX BACKTRACK FUNC DEF
#         def backtrack(i:int, j:int, k:str) -> bool:

#             if new_trie.search(k):
#                 return True
                    
#             if not new_trie.stars_with(k):
#                 return False
            
#             temp = board[i][j]
#             board[i][j] = '.'

#             #1
#             if 0<i<len(board)-1 and 0<j<len(board[0])-1:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True
            
#             #2
#             elif 0 == i and 0 == j:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i, j+1, k+board[i][j+1]):        
#                     return True
                
#             #3
#             elif 0 == i and 0<j<len(board[0])-1:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i, j+1, k+board[i][j+1]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True
            
#             #4
#             elif len(board)-1 == i and len(board[0])-1 == j:
#                 if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True
            
#             #5
#             elif 0<i<len(board)-1 and 0 == j:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]):        
#                     return True
                
#             #6
#             elif 0<i<len(board)-1 and len(board[0])-1 == j:
#                 if backtrack(i+1, j, k+board[i+1][j]) or backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True
            
#             #7
#             elif len(board)-1 == i and 0 == j:
#                 if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]):        
#                     return True
            
#             #8
#             elif len(board)-1 == i and 0<j<len(board[0])-1:
#                 if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j+1, k+board[i][j+1]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True

#             #9
#             elif len(board)-1 == i and len(board[0])-1 == j:
#                 if backtrack(i-1, j, k+board[i-1][j]) or backtrack(i, j-1, k+board[i][j-1]):        
#                     return True


#             board[i][j] = temp

#             return False 
        

#         # COLLECT FIRST LETTER COORDINATES FOR EACH WORD
#         words_dict = {}

#         for word in words:

#             coordinates = []

#             for i,row in enumerate(board):
#                 coordinates.extend([(i,j) for j,elem in enumerate(row) if board[i][j] == word[0]])

#             if coordinates:
#                 words_dict[word] = coordinates


#         # INITIATE THE TRIE
#         new_trie = Trie()

#         for word in words_dict.keys():
#             new_trie.insert(word)

#         x = 0

#         result = []

#         # ITERATE THE DICT
#         for word in words_dict:

#             temp_board = copy.deepcopy(board)

#             for i,j in words_dict[word]:

#                 if backtrack(i, j, word[0]):

#                     result.append(word)
#                     board = temp_board

#         return result

#     # Testing
#     print(findWords(board=board, words=words))

#     '''
#     Notes:
#         My solution and approach wasn't that far. The logic was correct, the execution was the one to fail.
#         My version of the solution tends to get redundant and can't handle efficiently larger inputs
#     '''

#     # TrieNode Definition
#     class TrieNode:

#         def __init__(self):
#             self.values = {}
#             self.is_word = False

#     # Trie DS Definition
#     class Trie:

#         def __init__(self):
#             self.root = TrieNode()
        
#         def insert(self, word:str) -> None:

#             curr_node = self.root

#             for char in word:
#                 if char not in curr_node.values:
#                     curr_node.values[char] = TrieNode()            
#                 curr_node = curr_node.values[char]
            
#             curr_node.is_word = True

#     'Actual Solution'
#     def findWords(board: list[list[str]], words: list[str]) -> list[str]:

#         # Build the Trie
#         trie = Trie()

#         for word in words:
#             trie.insert(word)
        
#         # Auxiliary vars
#         rows, cols = len(board), len(board[0])
#         result = set()
#         visited = set()

#         #Aux DFS Func
#         def dfs(node:TrieNode, i:int, j:str, path:str) -> None:

#             if i<0 or i>=rows or j<0 or j>=cols or (i,j) in visited or board[i][j] not in node.values:
#                 return
            
#             visited.add((i,j))
#             node = node.values[board[i][j]]
#             path += board[i][j]

#             if node.is_word:
#                 result.add(path)
#                 node.is_word = False    # To avoid duplicate results

#             # Explore neighbors in 4 directions (up, down, left, right)
#             for x, y in [(i-1,j),(i+1,j),(i,j-1),(i,j+1)]:
#                 dfs(node, x, y, path)
            
#             visited.remove((i,j))        

#         # Traverse the board
#         for i in range(rows):
#             for j in range(cols):
#                 dfs(trie.root, i, j, '')        

#         return result

#     'Done'

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

'''329. Longest Increasing Path in a Matrix'''
# def x():

#     # Input
#     # Case 1
#     matrix = [[9,9,4],[6,6,8],[2,1,1]]
#     # Output: 4 // Longest path [1, 2, 6, 9]

#     # Case 2
#     matrix = [[3,4,5],[3,2,6],[2,2,1]]
#     # Output: 4 // Longest path [3, 4, 5, 6]


#     '''
#     My Approach (DP)
    
#         Intuition:

#             Thinking in the matrix as a graph my intuition is to check each node
#             following DFS for its vecinity only if the neighbor is higher than the curr node value,
#             and store the possible path length from each node in a DP matrix. after traversing the graph
#             the max value in the DP matrix will be the answer.
#     '''

#     def longestIncreasingPath(matrix: list[list[int]]) -> int:

#         # Handle corner case: no matrix
#         if not matrix or not matrix[0]:
#             return 0

#         # Capturing the matrix dimentions
#         m,n = len(matrix), len(matrix[0])

#         # Defining the DP matrix
#         dp = [[1]*n for _ in range(m)]

#         # Define the directions for later adding the neighbors
#         directions = [(1,0),(-1,0),(0,1),(0,-1)]
        
#         # Traverse the matrix
#         for i in range(m):

#             for j in range(n):

#                 # Define its max: its current max path in the dp matrix
#                 elem_max = dp[i][j]

#                 # Define the actual neighbors: The element within the matrix boundaries and higher and itself
#                 neighbors = [(i+dx, j+dy) for dx,dy in directions if 0<=i+dx<m and 0<= j+dy<n and matrix[i+dx][j+dy] > matrix[i][j]]

#                 # Check for each neighbor's max path while redefine its own max path
#                 for neighbor in neighbors:
#                     curr = dp[i][j]
#                     next_max = max(curr, curr + dp[neighbor[0]][neighbor[1]])
#                     elem_max = max(elem_max, next_max)
                
#                 # Update it in the dp matrix
#                 dp[i][j] = elem_max    

#         # get dp's max
#         result = max(max(x) for x in dp)
        
#         # Return its value
#         return result

#     # Testing
#     print(longestIncreasingPath(matrix=matrix))

#     'Note: This approach only works if it starts from the node with the largest value'


#     'DFS with Memoization Approach'
#     def longestIncreasingPath(matrix: list[list[int]]) -> int:

#         # Handle Corner Case
#         if not matrix or not matrix[0]:
#             return 0

#         # Capture matrix's dimentions
#         m, n = len(matrix), len(matrix[0])

#         # Define the memoization table
#         dp = [[-1] * n for _ in range(m)]

#         # Define the directions
#         directions = [(1,0),(-1,0),(0,1),(0,-1)]
        
#         # Define the DFS helper function
#         def dfs(x, y):

#             # Handle corner case: the cell was already visited
#             if dp[x][y] != -1:
#                 return dp[x][y]
            
#             # Define the max starting path, which is 1 for any cell
#             max_path = 1

#             # Define the directions to go
#             for dx, dy in directions:

#                 nx, ny = x + dx, y + dy

#                 # If it's a valid neighbor, recalculate the path
#                 if 0 <= nx < m and 0 <= ny < n and matrix[nx][ny] > matrix[x][y]:
                    
#                     # The new path will be the max between the existing max path and any other valid path from the neighbor
#                     max_path = max(max_path, 1 + dfs(nx, ny))
            
#             # Update the Memoization table
#             dp[x][y] = max_path
            
#             # Return the value
#             return dp[x][y]
        

#         # Define the initial max lenght
#         max_len = 0

#         # Run the main loop for each cell
#         for i in range(m):
#             for j in range(n):
#                 max_len = max(max_len, dfs(i, j))
        
#         # Return the max length
#         return max_len

#     # Testing
#     print(longestIncreasingPath(matrix=matrix))

#     'Done'

'''341. Flatten Nested List Iterator'''
# def x():

#     # Base
#     """
#     This is the interface that allows for creating nested lists.
#     You should not implement it, or speculate about its implementation
#     """

#     class NestedInteger:
#        def isInteger(self) -> bool:
#            """
#            @return True if this NestedInteger holds a single integer, rather than a nested list.
#            """

#        def getInteger(self) -> int:
#            """
#            @return the single integer that this NestedInteger holds, if it holds a single integer
#            Return None if this NestedInteger holds a nested list
#            """

#        def getList(self) -> None: #[NestedInteger] is the actual expected return
#            """
#            @return the nested list that this NestedInteger holds, if it holds a nested list
#            Return None if this NestedInteger holds a single integer
#            """

#     # Input
#     # Case 1
#     nested_list = [[1,0],2,[1,1]]
#     # Output: [1,1,2,1,1]

#     # Case 2
#     nested_list = [1,[4,[6]]]
#     # Output: [1,4,6]


#     'The Solution'
#     class NestedIterator:

#         def __init__(self, nestedList: list[NestedInteger]):
        
#             # Initialize the stack with the reversed nested list
#             self.stack = nestedList[::-1]
        
#         def next(self) -> int:

#             # The next element must be an integer, just pop and return it
#             return self.stack.pop().getInteger()
        
#         def hasNext(self) -> bool:

#             # While there are elements in the stack and the top element is an Integer to be returned
#             while self.stack:
                
#                 # Peek at the top element
#                 top = self.stack[-1]
                
#                 # If it's an integer, we're done
#                 if top.isInteger():
#                     return True
                
#                 # Otherwise, it's a list, pop it and push its contents onto the stack
#                 self.stack.pop()
#                 self.stack.extend(top.getList()[::-1])
            
#             # If the stack is empty, return False
#             return False

#     'Done'




'''114. Flatten Binary Tree to Linked List'''
# def x():
    
#     from typing import Optional

#     # Definition for a binary tree node
#     class TreeNode:
#         def __init__(self, val=0, left=None, right=None):
#             self.val = val
#             self.left = left
#             self.right = right

#     # Input
#     # Case 1
#     tree = [1,2,5,3,4,None,6]
#     root = TreeNode(val=1, 
#                     left=TreeNode(val=2,
#                                   left=TreeNode(val=3),
#                                   right=TreeNode(val=4)),
#                     right=TreeNode(val=5,
#                                    right=TreeNode(val=6))
#                     )
#     # Output: [1,null,2,null,3,null,4,null,5,null,6]

#     '''
#     My Approach

#         Intuition:
            
#             - Handle Corner Case: No Node passed
           
#             - Create a 'dummy' head pointer into which the linked list will be built
#                 and a 'curr' pointer that will be located to in the 'right' pointer of the dummy.
            
#             - Define a preorder traversal function: 
#                 *This function will add each node to the curr's 'right' pointer.
#                 *And will also move the 'curr' pointer to the right to be located at the just added node.

#             - Reassign 'root' to the dummy's 'right' pointer
#     '''

#     'O(n) Approach'
#     def flatten(root: Optional[TreeNode]) -> None:

#         # Handle Corner case: ...
#         if not root:
#             return None
                
#         ll_layout = []

#         # Preorder traversal function definition
#         def preorder(node:TreeNode) -> None:

#             if not node:
#                 return    

#             ll_layout.append(node)
#             preorder(node=node.left)
#             preorder(node=node.right)
        

#         preorder(node=root)


#         for i in range(len(ll_layout)-1):

#             curr = ll_layout[i]
#             curr.left = None
#             curr.right = ll_layout[i+1]
               
    
#     # Testing
#     print(flatten(root=root))
        


#     'Optimized O(1) Space Solution'
#     def flatten(root: Optional[TreeNode]) -> None:

#         # Handle Corner case: No node passed
#         if not root:
#             return None
                
        
#         # Create a mutable container for curr so that changes are shared
#         curr = [None]  # Using a list to hold the current pointer
        

#         # Preorder traversal function definition
#         def preorder(node:TreeNode) -> None:

#             if not node:
#                 return
           
#             # Flatten the current node
#             if curr[0]:  # If curr[0] exists, link it to the current node
#                 curr[0].right = node
#                 curr[0].left = None

#             # Move curr to the current node
#             curr[0] = node
            
#             # Save the right subtree before recursion (because we modify the right pointer)
#             right_subtree = node.right
            
#             # Traverse left subtree first (preorder)
#             preorder(node.left)
            
#             # Traverse right subtree last
#             preorder(right_subtree)
        
#         # Traverse the root with the preorder function
#         preorder(node=root)


#     # Testing
#     print(flatten(root=root))
    

#     '''
#     Note: 
       
#        - Within a recursion inmutable object can not be affected out of the function, since Python creates a copy of them to work locally, the workaround here is to work with mutables (a list).

#        - The right sub-tree must be saved because is the next recursive call the right pointer will be modified and when it rolls back, the remaining right part will be lost otherwise.
    
#     '''

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

'''994. Rotting Oranges'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     grid = [[2,1,1],[1,1,0],[0,1,1]]
#     # Output: 4

#     # Case 2
#     grid = [[2,1,1],[0,1,1],[1,0,1]]
#     # Output: -1

#     # Case 3
#     grid = [[0,2]]
#     # Output: 0
    
#     # Custom Case
#     grid = [[1,2]]
#     # Output: 1

#     '''
#     My Approach (BFS)

#         Intuition:
            
#             - Flatten the input matrix to have the individual elements of it.
#             - Handle corner case: No fresh oranges, return 0
#             - Handle corner case: No rotten oranges, return -1

#             - Collect all fresh oranges with not adjacent neighbors.
#             - if there's any, Handle corner case: fresh orange with no adjacent neighbors, return -1.

#             - Initialize a 'minutes' counter in 0.
#             - Collect all rotten oranges coordinates in a queue 'nodes'.

#             - In a while loop [while 'nodes' exists]:
#                 + Create a list holder 'new_nodes'.
#                 + Pop the first element of nodes.
#                 + Append all its adjacent neighbors that are fresh oranges in 'neighbors'.
#                 + if 'nodes' is empty: add 1 to minutes and extend 'nodes' to 'neighbors.
            
#             - return 'minutes'
#     '''

#     def orangesRotting(grid: list[list[int]]) -> int:
        
#         # Capture grid dimentions
#         m,n = len(grid), len(grid[0])

#         # Flatten the input matrix
#         elems = [elem for row in grid for elem in row]

#         # Handle corner case: No fresh oranges
#         if 1 not in elems:
#             return 0

#         # Define coordinates differentials
#         diff = [(1,0),(-1,0),(0,1),(0,-1)]

#         # Collect all fresh oranges with rotten oranges as neighbors.  
#         for i in range(m):
#             for j in range(n):

#                 if grid[i][j]==1:  

#                     fresh = [grid[i+dx][j+dy] for dx,dy in diff if 0<=i+dx<m and 0<=j+dy<n]
                    
#                     if sum(fresh) == 0:
#                         return -1

#         # Initialize minutes counter in 0
#         minutes = 0

#         # Collect all rotten oranges coordinates
#         queue = [(i,j) for i in range(n) for j in range(m) if grid[i][j]==2]

#         # Define the newly rotten oranges holder
#         new_nodes = []  

#         while queue:

#             # Pop the first rotten orange in the queue
#             node = queue.pop(0)
                
#             # Collect each rotten oraange fresh orange neighbor
#             neighbors = [(node[0]+dx, node[1]+dy) for dx, dy in diff if 0<=node[0]+dx<m and 0<=node[1]+dy<n and grid[node[0]+dx][node[1]+dy]==1]

#             # Root each fresh orange neighbor
#             for elem in neighbors:
#                 grid[elem[0]][elem[1]] = 2
            
#             # Pass the neigbors into the new_nodes holders
#             new_nodes.extend(neighbors)

#             if not queue and new_nodes:
#                 # Add a new minute to the minute counter
#                 minutes += 1

#                 # Update the queue
#                 queue.extend(new_nodes)

#                 # Reset new_nodes
#                 new_nodes.clear()        
 
#         # Return minutes
#         return minutes

#     # Testing
#     print(orangesRotting(grid=grid))

#     '''Note: This solution only met 97% of test cases, it didn't handled correctly the remaining cases'''

#     '''
#     Corrected Solution
#     '''

#     def orangesRotting(self, grid: list[list[int]]) -> int:
        
#         from collections import deque

#         # Grid dimensions
#         m, n = len(grid), len(grid[0])

#         # Initialize queue with all initially rotten oranges and count fresh oranges
#         queue = deque()
#         fresh_count = 0

#         for i in range(m):
#             for j in range(n):
#                 if grid[i][j] == 2:
#                     queue.append((i, j))  # Rotten orange coordinates
#                 elif grid[i][j] == 1:
#                     fresh_count += 1  # Count of fresh oranges

#         # Edge case: No fresh oranges to begin with
#         if fresh_count == 0:
#             return 0

#         # Coordinate differentials for the four adjacent directions
#         directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

#         # BFS processing
#         minutes = 0

#         while queue:

#             # Process each level of BFS (each minute)
#             for _ in range(len(queue)):
                
#                 x, y = queue.popleft()

#                 # Rot adjacent fresh oranges
#                 for dx, dy in directions:

#                     nx, ny = x + dx, y + dy

#                     # Check bounds and if the orange is fresh
#                     if 0 <= nx < m and 0 <= ny < n and grid[nx][ny] == 1:
#                         grid[nx][ny] = 2  # Mark as rotten
#                         fresh_count -= 1  # Decrement fresh orange count
#                         queue.append((nx, ny))  # Add newly rotten orange to queue

#             # Increment minutes after processing each level
#             if queue:  # Only increment if there are still oranges rotting in the next minute
#                 minutes += 1

#         # If there are fresh oranges left, return -1, otherwise return minutes
#         return minutes if fresh_count == 0 else -1

'''100. Same Tree'''
# def x():
    
#     from typing import Optional

#     # Input
#     # Case 1
#     p = TreeNode(1, TreeNode(2), TreeNode(3))
#     q = TreeNode(1, TreeNode(2), TreeNode(3))
#     # Output: True

#     # Case 2
#     p = TreeNode(1, TreeNode(2))
#     q = TreeNode(1, None, TreeNode(2))
#     # Output: False

#     # Case 3
#     p = TreeNode(1, TreeNode(2), TreeNode(1))
#     q = TreeNode(1, TreeNode(1), TreeNode(2))
#     # Output: False

#     # Case 4
#     p = TreeNode(1, TreeNode(2), TreeNode(1))
#     q = TreeNode(1, TreeNode(2), TreeNode(1))
#     # Output: True

#     '''
#     My Approach

#         Intuition:
            
#             - Perform a DFS for both nodes at same time and compare node by node as
#                 the algorithm traverses the trees.
#     '''

#     def isSameTree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:

#         # Handle Corner case: No nodes passed (p & q)
#         if not p and not q:
#             return True
        
#         # Handle Corner case: One of the two nodes was not passed
#         if (p and not q) or (q and not p):
#             return False
        
#         # Initialize both stacks for the traversal
#         stack_p, stack_q = [p],[q]

#         # Process the trees
#         while stack_p and stack_q:

#             node_p, node_q = stack_p.pop(), stack_q.pop()

#             if node_p and node_q and node_p.val == node_q.val:

#                 stack_p.extend([node_p.left, node_p.right])
#                 stack_q.extend([node_q.left, node_q.right])
            
#             elif not node_p and not node_q:
#                 pass

#             else:
#                 return False
        
#         # Return True if the algorithm gets to this point
#         return True

#     # Testing
#     print(isSameTree(p=p, q=q))

#     '''Note: this solution where already rpetty solid but the recursive version is cleaner. Either way I prefer this one'''




#     '''
#     Recursive Approach       
#     '''

#     def isSameTree(p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:

#         # Both are None
#         if not p and not q:
#             return True
        
#         # Both are differente
#         if not p or not q or p.val != q.val:
#             return False        
        
#         # Make it recursive
#         return isSameTree(p=p.left,q=q.left) and isSameTree(p=p.right,q=q.right)

#     # Testing
#     print(isSameTree(p=p, q=q))

#     '''Note: Done'''



























