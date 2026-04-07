# Contributing to LLM Inference Engineering Course

Thank you for your interest in contributing to this educational resource! This document provides guidelines and instructions for contributing.

## 🎯 Contribution Philosophy

This course aims to be:
1. **Accurate**: Technically correct and up-to-date
2. **Accessible**: Understandable for the target audience
3. **Practical**: Focused on real-world implementation
4. **Comprehensive**: Covering both theory and practice

## 📋 Types of Contributions

### 1. Content Improvements
- Fixing technical errors
- Updating outdated information
- Improving explanations
- Adding missing concepts

### 2. Code Contributions
- Bug fixes in labs
- Performance improvements
- Additional examples
- Better test coverage

### 3. New Content
- Additional labs or exercises
- New lecture notes
- Case studies
- Interview questions

### 4. Documentation
- Better setup instructions
- Troubleshooting guides
- Translation to other languages
- Accessibility improvements

## 🚀 Getting Started

### Prerequisites
- Git installed
- Python 3.9+ environment
- Basic understanding of LLM inference

### Setup Development Environment
```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/llm-inference-engineering.git
cd llm-inference-engineering

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

### Branch Strategy
```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or a fix branch
git checkout -b fix/issue-description
```

## 📝 Contribution Guidelines

### Code Style
- Follow PEP 8 for Python code
- Use type hints where appropriate
- Include docstrings for all functions/classes
- Write meaningful commit messages

### Notebook Guidelines
- Clear markdown explanations
- Well-commented code cells
- Output cells showing results
- Minimal external dependencies in labs

### Documentation Standards
- Use clear, concise language
- Include examples where helpful
- Link to relevant resources
- Maintain consistent formatting

## 🔧 Development Workflow

### 1. Small Changes (Typo fixes, minor updates)
- Direct PR to main branch is acceptable
- Include brief description of changes

### 2. Medium Changes (New labs, major updates)
- Create an issue first to discuss
- Work on a feature branch
- Include tests where applicable
- Update documentation as needed

### 3. Large Changes (New modules, architectural changes)
- Start with a proposal in Issues
- Get feedback from maintainers
- Create a detailed implementation plan
- Consider backward compatibility

### Testing Your Changes
```bash
# Run basic tests
pytest tests/

# Check code style
black --check .
flake8 .
mypy .

# Test notebooks execute without errors
python scripts/test_notebooks.py
```

## 📚 Content Standards

### Technical Accuracy
- Cite sources for technical claims
- Include references to papers
- Verify code examples work
- Test on multiple environments

### Pedagogical Quality
- Start with simple examples
- Build complexity gradually
- Include practical applications
- Connect theory to practice

### Accessibility
- Use clear, simple language
- Include visual aids where helpful
- Provide multiple learning paths
- Consider different learning styles

## 📖 Documentation Structure

### Lecture Notes
- Place in appropriate week folder
- Use Markdown with MathJax for equations
- Include learning objectives
- Add references and further reading

### Labs
- Self-contained Jupyter notebooks
- Clear instructions and expected outcomes
- Solution notebooks (in solutions/ folder)
- Grading rubrics if applicable

### Code Examples
- Well-documented functions
- Unit tests
- Performance benchmarks
- Usage examples

## 🐛 Reporting Issues

### Bug Reports
Include:
1. Clear description of the issue
2. Steps to reproduce
3. Expected vs actual behavior
4. Environment details
5. Error messages or screenshots

### Feature Requests
Include:
1. Problem you're trying to solve
2. Proposed solution
3. Alternative solutions considered
4. Use cases or examples

### Content Suggestions
Include:
1. Topic to add/improve
2. Why it's important
3. Suggested approach
4. Relevant resources

## 🔍 Review Process

### What Reviewers Look For
1. **Technical correctness**: Is the information accurate?
2. **Clarity**: Is it easy to understand?
3. **Completeness**: Are all necessary details included?
4. **Consistency**: Does it match the course style?
5. **Practicality**: Will it help students learn?

### Review Timeline
- Small changes: 1-3 days
- Medium changes: 3-7 days  
- Large changes: 1-2 weeks

### Addressing Feedback
- Be responsive to reviewer comments
- Make requested changes or explain why not
- Keep discussions focused and constructive
- Thank reviewers for their time

## 📄 License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers the project.

## 🙏 Acknowledgments

All contributors will be acknowledged in:
- The README.md file
- Individual content files they contribute to
- Release notes

## 🎓 For Academic Contributors

If you're contributing academic content:
- Ensure you have rights to share the material
- Cite original sources appropriately
- Consider open educational resource licenses
- Maintain academic integrity standards

## 💬 Communication

- Use GitHub Issues for technical discussions
- Be respectful and constructive
- Assume good intent
- Help others learn and grow

## 🏆 Recognition

Significant contributions may lead to:
- Co-maintainer status
- Featured contributor spot
- Conference presentation opportunities
- Research collaboration opportunities

---

Thank you for helping make this course better for everyone! 🚀