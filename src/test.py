#!/usr/local/bin/python
from Model import Model
from Executor import Executor

model = Model()
model.assemble_graph()

silence_step = 0
skip_step = 20

exe = Executor(model, silence_step=silence_step, skip_step=skip_step)

exe.train_and_dev()
exe.restore_and_test()
