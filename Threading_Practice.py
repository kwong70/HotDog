import threading

def func(name):
    print(type(name))
    print(str(name) + "hello")

threads = []
for i in range(3):
    x = threading.Thread(target = func, args = (i, ) )
    x.start()
    threads.append(x)


for thread in threads:
    thread.join()
