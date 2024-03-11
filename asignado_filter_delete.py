class AsignadoFilter(admin.SimpleListFilter):
    _title = 'Asignación'
    _parameter_name = 'asignado'

    def lookups(self, request, model_admin):
        return (
            ('sí', 'sí'),
            ('no', 'no'),
        )

    def queryset(self, request, queryset):
        value = self.value()

    @property
    def title(self):
        return self._title
    
    @property
    def parameter_name(self)
        return self._parameter_name