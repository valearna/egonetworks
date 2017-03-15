.. doc page for the core.py file
   
egonetworks.core -- Basic ego network classes and functions
===========================================================

This is the main module of the package and contains the basic ego network models and functions for ego network analysis.

The main class of the module is the **abstract** class :class:`egonetworks.core.EgoNetwork`. This contains the
data structures to represent generic social relationship between people (specified by a list of alter identifiers and
the frequencies of contact between the ego and these alters). In addition, the class provides the main methods for the
static and dynamic analysis of ego networks. For example,
:any:`egonetworks.core.EgoNetwork.get_optimal_num_circles` returns the optimal number of ego network circles
of the ego network, and :any:`egonetworks.core.EgoNetwork.get_circles_properties` returns the properties of
the circles given their number a priori.

Note that :class:`egonetworks.core.EgoNetwork` cannot be instantiated directly. Its main implementation, within this
module, are :class:`egonetworks.core.FrequencyEgoNetwork` and :class:`egonetworks.core.ContactEgoNetwork`. The
former class implements methods to add relationships to the ego network by specifying contact frequencies directly,
whereas the latter implements methods to add single social contacts (i.e. interactions involving communication between
the ego and the alters such as text messages or online posts) to the ego network. In
:class:`egonetworks.core.ContactEgoNetwork`, contact frequencies are calculated automatically from social contacts.

.. autoclass:: egonetworks.core.ContactAdditionalInfo
   :members:

.. autoclass:: egonetworks.core.EgoNetwork
   :members:

.. autoclass:: egonetworks.core.ContactEgoNetwork
   :members:

.. autoclass:: egonetworks.core.FrequencyEgoNetwork
   :members:   

These are the definitions of the objects returned (or required as arguments) by some of the methods of the classes in
this module:

.. autoclass:: egonetworks.core.CirclesProperties
   :members:

.. autoclass:: egonetworks.core.AlterFreq
   :members:

.. autoclass:: egonetworks.core.BasicContact
   :members:

.. autoclass:: egonetworks.core.FullContact
   :members:

.. autoclass:: egonetworks.core.MinContactsTimeWin
   :members:

.. autoclass:: egonetworks.core.TopicIntensity
   :members:

.. autoclass:: egonetworks.core.RelationshipsProperties
   :members:

.. toctree::
   :maxdepth: 2
