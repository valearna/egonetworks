.. egonetworks documentation master file, created by
   sphinx-quickstart on Tue May  3 15:04:24 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

egonetworks -- Python package for Ego network structural analysis
=================================================================

This package contains classes and functions for the structural analysis of ego networks.

An ego network is a simple model that represents a social network from the point of view of an individual. This model
considers only the social relationships that a focal node in the network (termed *ego*) maintains with other nodes
(termed *alters*). Note that the model supported by this package does not consider relationships between alters (aka
mutual friendship relationships), but only the star topology of alters connected to the ego. This ego network model is
known as "Dunbar's ego network". See [1]_ and [2]_ for additional information about ego networks and ego network
analysis.

The package offers several methods for the static and dynamic analysis of ego networks. For example, the package
provides a function to obtain the "social circles" of the ego network, which are discrete groups of alters at similar
level of tie strength with the ego. In addition, there are functions to analyse the dynamic evolution of ego networks
and to calculate their stability over time. These functions are useful, for example, for the analysis of human behaviour
in different social environments as well as to identify particularly active, dynamic or sociable people from their
communication traces.

The package offers specialised classes for building and studying ego networks from Twitter data and from
coauthorship or collaboration networks (i.e. networks where the ego is an author and the alters are people with whom he
or she coauthored publications).

These are the main modules of the package:

.. toctree::
   :maxdepth: 2

   core	
   twitter
   coauthorship
   generic
   similarity
   error

References
----------
   
.. [1] R.I.M. Dunbar, V. Arnaboldi, M. Conti, A. Passarella, "`The Structure of Online Social Networks Mirrors Those in the Offline World <http://www.sciencedirect.com/science/article/pii/S0378873315000313>`_", Social Networks, Vol. 43, October 2015, Pages 39-47
       
.. [2] A. Valerio, A. Passarella, M. Conti, R.I.M. Dunbar, "`Online Social Networks: Human Cognitive Constraints in Facebook and Twitter Personal Graphs <http://www.sciencedirect.com/science/book/9780128030233>`_", A volume in Computer Science Reviews and Trends, Elsevier, ISBN: 978-0-12-803023-3, 2015
