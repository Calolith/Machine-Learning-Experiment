module ParseLangModel

export readData

function readData(path, word2idx, idx2word)
	f = open(path)
	lines = readlines(f)
	close(f)

	data = Array(Int, 0)
	for line in lines
		words = split(line)
		for word in words
			if !haskey(word2idx, word)
				word2idx[word] = length(word2idx) + 1
				idx2word[length(word2idx) + 1] = word
			end
			push!(data, word2idx[word])
		end
		push!(data, word2idx["<eos>"])
	end

	return data
end

end