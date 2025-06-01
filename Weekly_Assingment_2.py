class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:
    def __init__(self):
        self.head = None

    def add(self, data):
        new_node = Node(data)
        if self.head is None:
            self.head = new_node
        else:
            temp = self.head
            while temp.next:
                temp = temp.next
            temp.next = new_node

    def display(self):
        if self.head is None:
            print("List is empty!")
            return
        temp = self.head
        while temp:
            if temp.next:
                print(temp.data, end=' -> ')
            else:
                print(temp.data)
            temp = temp.next

    def delete_nth(self, n):
        if self.head is None:
            print("Nothing to delete, list is empty.")
            return

        if n <= 0:
            print(f"Invalid index: {n}")
            return

        temp = self.head

        if n == 1:
            self.head = temp.next
            return

        prev = None
        count = 1
        while temp and count < n:
            prev = temp
            temp = temp.next
            count += 1

        if temp is None:
            print(f"Index {n} is out of range.")
            return

        prev.next = temp.next


mylist = LinkedList()
mylist.add(100)
mylist.add(200)
mylist.add(300)
mylist.add(400)

print("Original list:")
mylist.display()

print("Deleting 3rd node:")
mylist.delete_nth(3)
mylist.display()

print("Deleting 1st node:")
mylist.delete_nth(1)
mylist.display()

print("Trying to delete 10th node:")
mylist.delete_nth(10)
mylist.display()

print("Clearing remaining nodes:")
mylist.delete_nth(1)
mylist.delete_nth(1)
mylist.display()

print("Trying to delete from empty list:")
mylist.delete_nth(1)
