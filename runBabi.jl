
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

	story_train, qstory_train = parseBabi(train_data_path, vocab)
	story_test, qstory_test = parseBabi(test_data_path, vocab)

	# Training and model details to be done
	# Right now, predictions are random.

	test_baseline(story_test, qstory_test, vocab)
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

function softmax_forw(W, b, data)
	xd = exp(W*data .+ b)
	return xd ./ sum(xd, 1)
end

function accuracy(ygold, yhat)
	correct = 0.0
	for i=1:size(ygold, 2)
		correct += indmax(ygold[:,i]) == indmax(yhat[:, i]) ? 1.0 : 0.0
	end
	return correct / size(ygold, 2)
end

function parseBabi(path, vocab)
	story = zeros(Int, 20, 15, 200)
	story_idx = 0
	sentence_idx = 0;
	qstory = zeros(Int, 20, 15, 200)

	open(path) do f
    	for line in enumerate(eachline(f))
    		is_question = false
    		if contains(line[2], "?")
    			is_question = true
    		end

    		arr = split(line[2])
    		sentence_idx = parse(Int, arr[1])
    		if sentence_idx == 1
    			story_idx = story_idx + 1
    		end

    		for i in 2:length(arr)
    			word = lowercase(arr[i])

    			if word[end] == '.' || word[end] == '?'
    				word = word[1:end-1]
    			end

    			if is_question
    				if haskey(vocab, word)
    					qstory[i - 1, sentence_idx, story_idx] = vocab[word]
    				end
    			else
    				story[i - 1, sentence_idx, story_idx] = vocab[word]
    			end

      		end
    	end
	end	
	return story, qstory
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

main()
