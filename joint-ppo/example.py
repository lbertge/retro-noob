from multiprocessing import JoinableQueue, Queue, Process
import time

class Worker(Process):
    def __init__(self, tasks, results, id):
        super().__init__()
        self.tasks = tasks
        self.results = results
        self.id = id
        self.completed_count = 0

    def run(self):
        while True:
            model = self.tasks.get()
            if model == 2:
                self.results.put("worker %d has completed %d tasks" % (self.id, self.completed_count))
                self.tasks.task_done()
                return
            print("model %s running from worker %d" % (str(model), self.id))
            time.sleep(1) # execution
            self.results.put("worker %d completed" % self.id)

            self.completed_count += 1
            self.tasks.task_done()

class Master():
    def __init__(self):
        self.tasks = JoinableQueue()
        self.results = Queue()

        self.workers = []
        for i in range(4):
            self.workers.append(Worker(self.tasks, self.results, i))

        for w in self.workers:
            w.start()

        steps = 4
        envs = 9
        for step in range(steps):
            for i in range(envs):
                self.tasks.put("params %d" % step)

            self.tasks.join() # block

            print("step %d completed" % step)

            while not self.results.empty():
                print(self.results.get())

        print("sending kill signal to workers")
        for w in self.workers:
            self.tasks.put(2)

        print("wrap up")
        self.tasks.join()
        while not self.results.empty():
            print(self.results.get())

if __name__ == '__main__':
    master = Master()
