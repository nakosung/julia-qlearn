module QLearn
	type Instance
		epsilon::Float32
		alpha::Float32
		gamma::Float32
		actions::Array
		q::Dict{Any,Float32}

		Instance(actions,eps=0.2,alpha=0.1,gamma=0.9) = new(eps,alpha,gamma,actions,Dict{Any,Float32}())
	end

	getQ(I::Instance, state, action, default_value = 0) = (k = (state,action); haskey(I.q,k) ? I.q[k] : default_value)

	function learnQ(I::Instance, state, action, reward, value) 
		oldv = getQ(I,state,action,())
		if oldv == ()
			I.q[(state,action)] = reward
		else
			I.q[(state,action)] = oldv + I.alpha * (value - oldv)
		end
	end

	function chooseAction(self::Instance, state)
		local action
		if rand() < self.epsilon
			action = self.actions[rand(1:end)]
		else
			q = [getQ(self,state,a) for a in self.actions]			
			maxQ,_ = findmax(q)
			c = count((v)->v==maxQ,q)
			local i
			if c > 1
				best = filter((k,v)->q[k] == maxQ,self.actions)
				i = best[rand(1:end)]
			else
				i = findfirst(q, maxQ)
			end
			action = self.actions[i]
		end
		action
	end

	function learn(self::Instance, state1, action1, reward, state2)
		maxqnew, _ = findmax([getQ(self,state2,a) for a in self.actions])
		learnQ(self,state1,action1,reward,reward + self.gamma * maxqnew)
	end

	function printQ(self::Instance)
		ks = collect(keys(self.q))
		states = Set([a for (a,b) in ks])
		actions = Set([b for (a,b) in ks])

		println(states)
		println(actions)
		println(self.q)
	end

	New = Instance

	export New, learn, chooseAction, printQ
end

learner = QLearn.New([:fwd,:bwd])

for iter = 1:100000
	QLearn.learn(learner, 0, :fwd, 0, 1)
	QLearn.learn(learner, 1, :fwd, 0, 2)
	QLearn.learn(learner, 1, :bwd, 0, 0)
	QLearn.learn(learner, 2, :bwd, 0, 1)
	QLearn.learn(learner, 0, :bwd, -5, -1)
	QLearn.learn(learner, 2, :fwd, 10, 3)
	QLearn.learn(learner, 3, :bwd, 0, 0)
end

QLearn.printQ(learner)

learner.epsilon = 0
for state in [0,1,2,3]
	println(state,"-->",QLearn.chooseAction(learner,state)) 
end