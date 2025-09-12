#include <iostream>
#include <string>
#include <vector>
#include <unicode/regex.h>


int main() {
    icu_77::UnicodeString s = "Hello, world! This is a test. This is another test.";

    UErrorCode status = U_ZERO_ERROR;
    icu_77::RegexMatcher m(R"('(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+)", s, 0, status);
    std::vector<std::string> matches;

    while (m.find(status) && U_SUCCESS(status)) {
        icu_77::UnicodeString us = m.group(status);
        std::string utf8;
        us.toUTF8String(utf8);
        matches.emplace_back(std::move(utf8));
    }

    for (const auto &w : matches) {
        std::cout << "Match: " << w << std::endl;
    }
}
