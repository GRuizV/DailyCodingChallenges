from typing import Optional, List

'''
CHALLENGES TAGS

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

'''

#Template
"xxx"
"""xxx"""
def x():
    
    # Input
    # Case 1
    head = None
    # Output: None

    '''
    My Approach

        Intuition:
            
            -...
    '''

    def y() -> int:

        # Handle Corner case: ...
        if not head:
            return
                
        # Return ...
        return 

    # Testing
    print(y())

    '''Note: Done'''


# Binary Tree Node Definition
class TreeNode:
    def __init__(self, val=0, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next

# List Node Definition
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

# Pretty Print Function
def pretty_print_bst(node:TreeNode, prefix="", is_left=True):

    if not node:
        return

    if node.right is not None:
        pretty_print_bst(node.right, prefix + ("│   " if is_left else "    "), False)

    print(prefix + ("└── " if is_left else "┌── ") + str(node.val))

    if node.left is not None:
        pretty_print_bst(node.left, prefix + ("    " if is_left else "│   "), True)








# Code Execution
x()






















































