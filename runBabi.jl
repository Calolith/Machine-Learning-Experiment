using Knet,AutoGrad,ArgParse,Compat

function main(args=ARGS)
	# Data path
	base_dir = "data/tasks_1-20_v1-1/en/"
	# Which data to train/test with
	task_id = 1 

	train_data_path = base_dir * searchdir(task_id, false, base_dir)
	test_data_path = base_dir * searchdir(task_id, true, base_dir)
	
	vocab = Dict{String,Int}()
	createVocab(vocab, train_data_path)
	createVocab(vocab, test_data_path)

	story_trn, qstory_trn, questions_trn = parseBabi(train_data_path, vocab)
	story_tst, qstory_tst, quesitons_tst = parseBabi(test_data_path, vocab)

	embed_A = randn(20, length(vocab)) * 0.3
	embed_B = randn(20, length(vocab)) * 0.3
	embed_C = randn(20, length(vocab)) * 0.3
	W = randn(length(vocab), 20) * 0.3
	param = Dict()
	param[1] = embed_A
	param[2] = embed_B
	param[3] = embed_C
	param[4] = W

	# Training
	losses = map(q->loss(param, questions_trn, story_trn, qstory_trn, vocab, q), 1:size(questions_trn, 2)) 
	accs = map(q->accuracy(param, questions_trn, story_trn, qstory_trn, vocab, q), 1:size(questions_trn, 2))
    println((:epoch,0,:loss, sum(losses) / size(questions_trn, 2), :acc, sum(accs) / size(questions_trn, 2)))
	for epoch in 1:50
		@time train(param, questions_trn, story_trn, qstory_trn, vocab)
		losses = map(q->loss(param, questions_trn, story_trn, qstory_trn, vocab, q), 1:size(questions_trn, 2)) 
    	accs = map(q->accuracy(param, questions_trn, story_trn, qstory_trn, vocab, q), 1:size(questions_trn, 2)) 
    	println((:epoch,epoch,:loss, sum(losses) / size(questions_trn, 2), :acc, sum(accs) / size(questions_trn, 2)))
	end
end

function train(param, questions, story, qstory, vocab; lr = 0.01)
	for q_num in 1:size(questions, 2)
		gloss = lossgradient(param, questions, story, qstory, vocab, q_num)
		for i in 1:4
			param[i] = param[i] - lr * gloss[i]	
		end
	end
end

function loss(param, questions, story, qstory, vocab, q_num)
	total = 0.0
	info = questions[:, q_num]
	M = createEmbedMatrix(param[1], story, info[2], info[3], vocab)
	u = bag_of_words(param[2], qstory[:, q_num], vocab)
	p = softmax_forw(transpose(u) * M, 2)
	C = createEmbedMatrix(param[3], story, info[2], info[3], vocab)
	o = sum(p .* C)
	yhat = softmax_forw(param[4] * (o + u), 1)
	
	ygold = zeros(Int, length(vocab))
	ygold[info[1]] = 1

	total += -sum(ygold .* log(yhat))
	return total
end

lossgradient = grad(loss)

function createEmbedMatrix(matrix, story, max_sentences, story_num, vocab)
	result = bag_of_words(matrix, story[:, 1, story_num], vocab)
	for j in 2:max_sentences
		result = hcat([result, bag_of_words(matrix, story[:, j, story_num], vocab)]...)
	end

#	result = zeros(20, max_sentences)
#	for j in 1:max_sentences
#		result[:, j] = bag_of_words(matrix, story[:, j, story_num], vocab)
		#result[:, j] = AutoGrad.getval(bag_of_words(matrix, story[:, j, story_num], vocab))
#	end
	return result
end

function bag_of_words(matrix, sentence, vocab)
	sentence_ohv = zeros(Int, length(vocab), length(sentence))
	for word_idx in 1:length(sentence)
		element = sentence[word_idx]
		if element != 0
			sentence_ohv[element, word_idx] = 1
		end
	end

	return sum(matrix * sentence_ohv, 2)[:]
end

function softmax_forw(matrix, side)
	xd = exp(matrix)
	return xd ./ sum(xd, side)
end

function accuracy(param, questions, story, qstory, vocab, q_num)
	total = 0.0
	info = questions[:, q_num]
	M = createEmbedMatrix(param[1], story, info[2], info[3], vocab)
	u = bag_of_words(param[2], qstory[:, q_num], vocab)
	p = softmax_forw(transpose(u) * M, 2)
	C = createEmbedMatrix(param[3], story, info[2], info[3], vocab)
	o = sum(p .* C)
	yhat = softmax_forw(param[4] * (o + u), 1)
	
	ygold = zeros(Int, length(vocab))
	ygold[info[1]] = 1

	total += indmax(ygold[:,1]) == indmax(yhat[:, 1]) ? 1.0 : 0.0
	return total

#	correct = 0.0
#	for i=1:size(ygold, 2)
#		correct += indmax(ygold[:,i]) == indmax(yhat[:, i]) ? 1.0 : 0.0
#	end
#	return correct / size(ygold, 2)
end

function parseBabi(path, vocab)
	story = zeros(Int, 20, 1000, 1000) # (word, sentence, story)
	qstory = zeros(Int, 20, 1000) # (word, question_idx)
	questions = zeros(Int, 3, 1000) # (info, question_idx), info = (answer, sentences_before, story)
	story_idx = 0
	sentence_idx = 0
	question_idx = 0
	max_words = 0
	max_sentences = 0

	open(path) do f
    	for line in enumerate(eachline(f))
    		is_question = false
    		if contains(line[2], "?")
    			is_question = true
    			question_idx = question_idx + 1
    		else
    			sentence_idx = sentence_idx + 1
    		end

    		arr = split(line[2])
    		if parse(Int, arr[1]) == 1
    			story_idx = story_idx + 1
    			if max_sentences < sentence_idx
    				max_sentences = sentence_idx
    			end
    			sentence_idx = 1
    		end

    		for i in 2:length(arr)
    			word = lowercase(arr[i])
    			answer_next = false
    			if word[end] == '.' || word[end] == '?'
    				answer_next = true
    				word = word[1:end-1]
    			end

    			if (i - 1 > max_words)
    				max_words = i - 1
    			end

    			if is_question
    				qstory[i - 1, question_idx] = vocab[word]
    				if answer_next
    					questions[1, question_idx] = vocab[lowercase(arr[i + 1])]
    					questions[2, question_idx] = sentence_idx
    					questions[3, question_idx] = story_idx
    					break
    				end
    			else
    				story[i - 1, sentence_idx, story_idx] = vocab[word]
    			end

      		end
    	end
	end
	story = story[1:max_words, 1:max_sentences, 1:story_idx]
	qstory = qstory[1:max_words, 1:question_idx]
	questions = questions[:, 1:question_idx]
	return story, qstory, questions
end

function searchdir(task_id, isTest, base_dir)
	key = "qa" * string(task_id) * "_"
	if isTest
		return filter(x->(contains(x, key) && contains(x, "test")), readdir(base_dir))[1]
	else
		return filter(x->(contains(x, key) && contains(x, "train")), readdir(base_dir))[1]
	end
end


function createVocab(vocab, path)
	open(path) do f
    	for line in enumerate(eachline(f))
    		arr = split(line[2])
    		for i in 2:length(arr)
    			word = lowercase(arr[i])
    			if word[end] == '.'
    				word = word[1:end-1]
    				get!(vocab, word, length(vocab)+1)

    			elseif word[end] == '?'
    				word = word[1:end-1]
    				get!(vocab, word, length(vocab)+1)
    				get!(vocab, arr[i + 1], length(vocab)+1)
    				break
    			else
    				get!(vocab, word, length(vocab)+1)
    			end
      		end
    	end
	end
end

function test_baseline(story, qstory, vocab)
	pred = zeros(Int, length(vocab), 1000)
	ygold = zeros(Int, length(vocab), 1000)
	j = 1
	for story_idx in 1:size(story, 3)
		for sentence_idx in 1:size(story, 2)
			if story[1, sentence_idx, story_idx] == 0
				i = 1
				while qstory[i + 1] != 0
					i = i + 1
				end
				ygold[qstory[i, sentence_idx, story_idx], j] = 1
				pred[rand(1:length(vocab)), j] = 1
				j = j + 1
			end
		end
	end
	acc = accuracy(ygold, pred)
	@printf("Vocab Size: %d Expected Accuracy: %g Accuracy: %g\n", length(vocab), 1.0/length(vocab), acc)
end

main()
