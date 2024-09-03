'''
CHALLENGES INDEX

98. Validate Binary Search Tree (Tree) (DFS)
101. Symmetric Tree (Tree) (BFS) (DFS)
102. Binary Tree Level Order Traversal (Tree) (BFS) (DFS)
103. Binary Tree Zigzag Level Order Traversal (BFS) (DFS)
104. Maximum Depth of Binary Tree (Tree) (BFS) (DFS)
116. Populating Next Right Pointers in Each Node (BFS) (DFS) (Tree)
124. Binary Tree Maximum Path Sum (DP) (Tree) (DFS)
127. Word Ladder (Hast Table) (BFS)
130. Surrounded Regions (Matrix) (BFS) (DFS)


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





























