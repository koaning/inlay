clean:
	rm testfoo*.vocab
	rm testfoo*.model

test:
	pytest

check: test clean
