'''
24. Swap Nodes in Pairs (LL)

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
'''24. Swap Nodes in Pairs'''
def x():

    from typing import Optional

    # Base
    # Definition for singly-linked list.
    class ListNode:
        def __init__(self, val=0, next=None):
            self.val = val
            self.next = next

    # Input
    # Case 1
    head = ListNode(1, next=ListNode(2,next=ListNode(3,next=ListNode(4))))
    # Output: [2,1,4,3]

    '''
    Solution

        Explanation:
            
            The key here is not the swapping itself but the notion of having a node before the 'first' and 'second' nodes to
            swap, since that node is the one that will connect back to 'second' and reconnecting the list.

            This happens at first in curr = dummy, that is a dummy node with value 0 that goes before the actual LL, that way
            is possible to make the swap with 'first' connecting to the node right next to 'second' and connecting 'second' to 'first',
            the crucial part is just after that happens, when curr connects back to 'second', that way the list makes sense again.
    '''

    def swapPairs(head: Optional[ListNode]) -> Optional[ListNode]:

        # Handle Corner case: no input.
        if not head:
            return
        
        # Handle Corner case: 1-element LL.
        if not head.next:
            return head
        
        # Create a dummy node to ease the head operations
        dummy = ListNode(0, next=head)

        # Initialize the current pointer
        curr = dummy

        # Start altering the list while traversing it
        while curr.next and curr.next.next:

            # Nodes to be swapped
            first = curr.next
            second = curr.next.next

            # Swapping the nodes
            first.next = second.next
            second.next = first
            curr.next = second

            # Move the current pointer two nodes ahead
            curr = first
        
        # Return the new head
        return dummy.next

    # Testing
    print(swapPairs(head=head))

    '''Note: Done'''

























































