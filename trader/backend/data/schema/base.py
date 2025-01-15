from ...imports import *


class DecBaseMixin:
    # Metadata ############################################################################################################################
    __table_args__ = {}

    __parent_relationship_kwargs__ = {
        'cascade': 'all, delete',
        'passive_deletes': True,
        'lazy': 'selectin'
    }

    __child_relationship_kwargs__ = {
        'passive_deletes': True,
        'lazy': 'selectin'
    }

    # Constructor ###########################################################################################################################
    def __init__(self, *args, **kwargs):
        # Set primary key id right off the bat so that it is accessible before the session commit
        setattr(self, f'{self.__tablename__}_id', generate_uuid())
        super().__init__(*args, **kwargs)

    # Representation ########################################################################################################################
    def __repr__(self):
        """
        Create a representation string of the Table Object (ORM row)
        :return: str
        """
        truncate = 100
        max_key_len = 0
        joins = []
        for k in self.__table__.columns.keys():
            if len(k) > max_key_len:
                max_key_len = len(k)
            attr = getattr(self, k)
            if isinstance(attr, str) and len(attr) > truncate:
                joins.append((k, f'{attr[:truncate // 2]}...{attr[-truncate // 2:]}'))
            elif isinstance(attr, (ogr.Geometry, dict)):
                s_attr = str(attr)
                if len(s_attr) > truncate:
                    joins.append((k, f'{s_attr[:truncate // 2]}...{s_attr[-truncate // 2:]}'))
                else:
                    joins.append((k, f'{s_attr}'))
            else:
                joins.append((k, attr))
        joins = '\n\t'.join([f'{i[0]:{max_key_len + 4}}{i[1]}' for i in joins])
        return f'<{self.__table__.name} ({self.__class__.__name__})\n\t' + joins + '\n>'

    # Class Methods ########################################################################################################################
    @classmethod
    def get_columns(cls, exclude_columns: list = []) -> list:
        """
        Returns a list of the defined column names for the ORM class
        """
        return [c.name for c in cls.__table__.columns if c.name not in exclude_columns]
    
    @classmethod
    def get_fields(
            cls, 
            only_fields: list = [],
            exclude_fields: list = [],
            exclude_id_fields: bool = False,
            include_properties: bool = True, 
            exclude_relationships: bool = True,
            exclude_last_edit: bool = False
    ) -> dict:
        """
        This is similar to the get_defined_vars() method except that it returns the field object
        instead of the value of the field.
        """
        if only_fields:
            final_fields = {k: v for k, v in vars(cls).items() if k in only_fields and k not in exclude_fields}
        else:
            relationships = cls.get_relationships()
            cls_vars = vars(cls)
            final_fields = {}
            for k, v in cls_vars.items():
                v_cls_name = v.__class__.__name__
                if k.startswith('_') or (exclude_id_fields and k.endswith('_id')) or (k in exclude_fields) or (exclude_last_edit and k == 'last_edit'):
                    continue

                if isinstance(v, property):
                    if include_properties:
                        final_fields[k] = v
                    else:
                        continue                
                if callable(v) or (exclude_relationships and k in relationships) or v_cls_name in ['classmethod', 'staticmethod']:
                    continue
                final_fields[k] = v
        return final_fields
    
    @classmethod
    def get_primary_key_field(cls) -> str:
        """
        Returns the primary key field name for a given ORM class
        """
        return cls.__table__.primary_key.columns.keys()[0]
    
    @classmethod
    def get_relationships(cls) -> list:
        """
        Returns a list of relationship keys for the class
        """
        return [r.key for r in cls.__mapper__.relationships]
    
    # Instance Methods ####################################################################################################################
    def get_primary_key(self) -> str:
        """
        Returns the primary key for a given ORM object instance,
        this can be useful when the primary key field name is not known.
        """
        return getattr(self, self.get_primary_key_field())
    
    def get_defined_vars(
            self, 
            only_fields: list = [],
            exclude_fields: list = [],
            exclude_id_fields: bool = True,
            include_properties: bool = True, 
            exclude_relationships: bool = True,
            exclude_last_edit: bool = True,
            other_attributes: list = []
    ) -> dict:
        """
        Gets the schema-defined attributes and values for an ORM object instance.
        This is a quick way to just get the defined attributes and ignore the SQLAlchemy overhead.
        """
        fields = self.get_fields(
            only_fields=only_fields,
            exclude_fields=exclude_fields,
            exclude_id_fields=exclude_id_fields,
            include_properties=include_properties,
            exclude_relationships=exclude_relationships,
            exclude_last_edit=exclude_last_edit
        )
        final_vars = {}
        for field in fields:
            final_vars[field] = getattr(self, field)

        for other_attr in other_attributes:
            if hasattr(self, other_attr) and other_attr not in final_vars:
                final_vars[other_attr] = getattr(self, other_attr)
        return final_vars

    def recordify(
            self,
            only_fields: list = [],
            primary_key_is_recid: bool = True,
            additional_fields: list = [],
            exclude_fields: list = [],
            map_attributes_modifier_funcs: dict = {}
    ) -> dict:
        """
        Prepares a ORM object instance as a record to be used by W2UI.        

        Previously this method was in the LocalManager class of the app (mbgt2_app.utils.local_manager.LocalManager) but it seems
        more appropriate to have it here in the mixin class which is inherited by the ORM schema subclasses - Changed 09/13/2024 - Zach Beebe.
        """
        record = {}
        primary_key_field = self.get_primary_key_field()
        vars_dict = self.get_defined_vars(
            only_fields=only_fields,
            exclude_fields=exclude_fields,
            exclude_id_fields=False,
            include_properties=False,
            exclude_relationships=True,
            exclude_last_edit=True,
            other_attributes=additional_fields
        )

        for key, value in vars_dict.items():
            if key == primary_key_field and primary_key_is_recid:
                record['recid'] = value
            else:
                record[key] = convert_numpy_value(value)         
            
            if key in map_attributes_modifier_funcs:
                record[key] = convert_numpy_value(map_attributes_modifier_funcs[key](record[key]))
        return record
    


###########################################################################################################################################
# The Base Class ###########################################################################################################################
# This will be inhertited by all ORM classes to provide the functionality of the DecBaseMixin
class Base(DecBaseMixin, DeclarativeBase):
    # Overwrite Constructor ###############################################################################################################
    def __init__(self, *args, **kwargs):
        # Different than inherited in the way that it does not automatically set the primary key id
        super().__init__(*args, **kwargs)

from .log import Log
from .user import User
from .alert import Alert
from .ticker import Ticker
from .trade import Trade
from .interval import Interval
from .candle import Candle