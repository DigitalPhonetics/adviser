# Home

## Introduction

!!! todo
    Add introduction

## Guiding Principles

### Modularity
In contrast to a traditional (rather static) pipeline approach which adheres to a fixed order of information flow, Adviser is implemented in an asynchronous way, using the publish-subscribe software pattern. This allows for parallel information flow which facilitates the combination of multiple modalities as well as the integration of additional modules.
For each module in a classic dialog system (NLU, BST, dialog policy and NLG), we provide a handcrafted baseline module, additionally we provide a reinforcement learning based implementation for the  policy. These can be used to quickly assemble a working dialog system or as implementation guidelines for custom modules. Additionally, because all modules inherit from the same abstract class, technical users can also easily write their own implementations or combinations of modules.

### Flexibility
The publish-subscribe pattern allows great flexibility in terms of structure and scope of the dialog system. Users can easily realize anything from a simple text-based pipeline system to a full-fledged multimodal, multi-domain dialog system.
Further, distributed systems are possible. Services are location-transparent and may thus be distributed across multiple machines. The central dialog system discovers local and remote services and provides synchronization guarantees for dialog initialization and termination. This is useful for resource-heavy tasks such as speech synthesis.

### Transparency
We provide a utility to draw the dialog graph, showing the information flow between services and any inconsistencies in publish/subscribe connections.

!!! todo
    Graphen einfügen?

### User-friendly at different levels
technical users have the full flexibility to explore and extend the back-end; non-technical users can use the provided code base for building systems; students from different disciplines could easily learn the concepts and explore human machine interaction.


## Support
You can ask questions by sending emails to <adviser-support@ims.uni-stuttgart.de>.

You can also post bug reports and feature requests in GitHub issues.

## How To Cite
If you use or reimplement any of this source code, please cite the following paper:

!!! todo
    Update reference

```bibtex
@InProceedings{
title =     {ADVISER: A Toolkit for DevelopingMulti-modal,Multi-domainandSocially-engagedConversational Agents},
author =    {Daniel Ortega and Dirk V{\"{a}}th and Gianna Weber and Lindsey Vanderlyn and Maximilian Schmidt and Moritz V{\"{o}}lkel and Zorica Karacevic and Ngoc Thang Vu},
booktitle = {Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019) - System Demonstrations},
publisher = {Association for Computational Linguistics},
location =  {Seattle, Washington, USA},
year =      {2020}
}
```


## License
Adviser is published under the <a href="https://www.gnu.org/licenses/gpl-3.0.de.html" target="_blank">GNU GPL 3</a> license.